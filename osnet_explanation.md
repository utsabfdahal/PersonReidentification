# How OSNet Works in This Project

## What Is OSNet?

**OSNet** (Omni-Scale Network) is a lightweight convolutional neural network designed specifically for **person Re-Identification (ReID)**. It was developed by Zhou et al. (2019) and is the core model that allows this project to recognise the same person across different video frames, camera angles, and lighting conditions.

The variant used here is **OSNet x1_0** — the full-scale version with ~2.2 million parameters, producing **512-dimensional embedding vectors**.

---

## The Core Idea

OSNet converts a cropped image of a person into a **512-number fingerprint** (called an embedding vector). Two images of the **same person** will produce similar vectors. Two images of **different people** will produce dissimilar vectors.

```
┌──────────────────┐         ┌──────────────────┐
│  Reference Photo │         │  Video Frame Crop │
│  of Target Person│         │  of Some Person   │
└────────┬─────────┘         └────────┬──────────┘
         │                            │
         ▼                            ▼
   ┌───────────┐                ┌───────────┐
   │   OSNet   │                │   OSNet   │
   │  (same    │                │  (same    │
   │   model)  │                │   model)  │
   └─────┬─────┘                └─────┬─────┘
         │                            │
         ▼                            ▼
  [0.12, -0.45, 0.78, ...]    [0.11, -0.43, 0.80, ...]
         512 numbers                  512 numbers
         │                            │
         └────────────┬───────────────┘
                      │
                      ▼
              Cosine Similarity
               = 0.94 (high!)
                      │
                      ▼
                "Same person!" ✅
```

---

## Why "Omni-Scale"?

Traditional CNNs look at features at a fixed scale — either small local patterns (texture of clothing) or large global patterns (body shape). **OSNet captures features at multiple scales simultaneously** through its unique building block:

### Omni-Scale Residual Block

```
Input Feature Map
       │
       ├──→ 1×1 conv (point-wise) ──→ captures very local features
       │
       ├──→ 3×3 conv (1 layer) ──→ captures small-scale features
       │
       ├──→ 3×3 conv (2 layers) ──→ captures medium-scale features
       │
       ├──→ 3×3 conv (3 layers) ──→ captures large-scale features
       │
       └──→ ...
              │
              ▼
       Aggregation Gate (learns which scales matter)
              │
              ▼
       Combined multi-scale features
```

The **Aggregation Gate** is a learned mechanism that dynamically weights which scales are most informative for each input image. For example:
- A person with a distinctive logo on their shirt → the network emphasises **small-scale** features (logo details).
- A person with a unique body silhouette → the network emphasises **large-scale** features (overall shape).

This makes OSNet robust across different scenarios without manual tuning.

---

## How This Project Uses OSNet (Step by Step)

### Step 1: Model Loading

```python
model = torchreid.models.build_model(
    name="osnet_x1_0",     # full-scale OSNet
    num_classes=1000,       # pretrained classification head (not used for ReID)
    pretrained=True,        # download ImageNet + ReID pretrained weights
)
model.eval()               # set to inference mode (no dropout, frozen batch norm)
```

The model is loaded with pretrained weights from **torchreid** — these weights were trained on large person ReID datasets (Market-1501, DukeMTMC-reID, MSMT17) so the network already understands how to distinguish between different people.

### Step 2: Preprocessing a Person Crop

Before feeding an image to OSNet, it must be standardised:

```python
def _preprocess(self, crop_bgr):
    # 1. Resize to fixed dimensions: 256 pixels tall, 128 pixels wide
    img = cv2.resize(crop_bgr, (128, 256))

    # 2. Convert BGR (OpenCV format) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Normalise pixel values from [0, 255] to [0.0, 1.0]
    img = img.astype(np.float32) / 255.0

    # 4. Apply ImageNet standardisation
    #    (subtract mean, divide by std — same preprocessing used during training)
    mean = [0.485, 0.456, 0.406]  # R, G, B channel means
    std  = [0.229, 0.224, 0.225]  # R, G, B channel stds
    img = (img - mean) / std

    # 5. Rearrange from (H, W, C) to (C, H, W) for PyTorch
    # 6. Add batch dimension: (C, H, W) → (1, C, H, W)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
```

**Why 256×128?** Person images are typically taller than they are wide. This 2:1 aspect ratio matches how people appear in surveillance footage and is the standard input size OSNet was trained on.

**Why ImageNet normalisation?** OSNet's backbone was initially pretrained on ImageNet, which uses these specific mean/std values. Using the same normalisation at inference time ensures the input distribution matches what the network expects.

### Step 3: Extracting the Embedding

```python
@torch.no_grad()  # no need to compute gradients during inference
def extract_embedding(self, crop_bgr):
    tensor = self._preprocess(crop_bgr).to(self.device)  # move to GPU/MPS
    feat = self.model(tensor)                              # forward pass → 512-d vector
    feat = F.normalize(feat, p=2, dim=1)                   # L2 normalise to unit length
    return feat.cpu().numpy().flatten()                     # → numpy array of 512 floats
```

The network's forward pass works like this internally:

```
Input Image (3 × 256 × 128)
        │
        ▼
  Conv Layer 1 (64 channels, stride 2) → 128 × 64
        │
        ▼
  Max Pool → 64 × 32
        │
        ▼
  OSNet Block 1 (256 channels) ← omni-scale features
        │
  Transition 1 → downsample
        │
        ▼
  OSNet Block 2 (384 channels) ← omni-scale features
        │
  Transition 2 → downsample
        │
        ▼
  OSNet Block 3 (512 channels) ← omni-scale features
        │
        ▼
  Global Average Pooling → 512 × 1 × 1
        │
        ▼
  Flatten → 512-dimensional vector
        │
        ▼
  L2 Normalisation → unit vector on 512-d hypersphere
```

The final **512 numbers** encode everything the network has learned about this person's appearance: clothing colour, texture patterns, body proportions, accessories, hair, etc.

**L2 normalisation** ensures every embedding lies on the surface of a unit hypersphere. This makes cosine similarity equivalent to a simple dot product, and ensures that the magnitude of the vector doesn't affect comparisons — only the direction matters.

### Step 4: Building Gold-Standard Reference Embeddings

For each reference photo, the project doesn't just compute one embedding — it creates a **robust average**:

```
Reference Photo
       │
       ▼
  YOLO: detect & crop the person
       │
       ▼
  SAM: remove background (fill with ImageNet mean)
       │
       ▼
  Generate 5 augmented variants:
       │
       ├── Original crop
       ├── Horizontal flip (mirror)
       ├── Brightness × 0.85 (darker)
       ├── Brightness × 1.15 (brighter)
       └── Centre crop (10% margins trimmed)
       │
       ▼
  OSNet: extract embedding for each variant
       │
       ▼
  Average all 5 embeddings → L2 normalise
       │
       ▼
  "Gold Standard" embedding for this reference
```

**Why augmentation?** The target person in the video may appear:
- Facing the opposite direction → horizontal flip handles this
- In shadow or bright sunlight → brightness variants handle this
- At different distances (tighter/wider framing) → centre crop handles this

By averaging embeddings across these variations, the gold standard becomes more tolerant of real-world appearance changes.

### Step 5: Matching During Video Processing

For every person detected in every video frame:

```python
# 1. Crop the person from the frame
crop = frame[y1:y2, x1:x2]

# 2. Extract their embedding
embedding = reid.extract_embedding(crop)  # → 512-d vector

# 3. Compare against all reference embeddings
for name, gold in gold_embeddings.items():
    similarity = cosine_similarity(embedding, gold)
```

**Cosine similarity** measures how similar two vectors are:

$$\text{cosine\_similarity}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

Since both vectors are already L2-normalised, this simplifies to a dot product:

$$\text{similarity} = a \cdot b = \sum_{i=1}^{512} a_i \times b_i$$

| Similarity | Interpretation |
|------------|----------------|
| 1.0 | Identical (same image) |
| 0.85 – 0.99 | Very likely the same person |
| **0.70 – 0.85** | **Probably the same person (our threshold zone)** |
| 0.50 – 0.70 | Uncertain — could be similar clothing |
| < 0.50 | Different person |

The project uses a threshold of **0.70** — if the best similarity exceeds this, the detection counts as a positive match for voting.

### Step 6: Majority Voting Confirmation

A single high-similarity frame isn't enough. The system requires **6 out of the last 10 frames** for a track ID to show a positive match before confirming it as the POI:

```
Frame:  1   2   3   4   5   6   7   8   9   10  11  12
Match:  ✓   ✗   ✓   ✓   ✗   ✓   ✓   ✓   ✗   ✓   ✓   ✓
                                              ↑
                                    Window [3-12]: 8/10 ✓
                                    8 ≥ 6 → CONFIRMED ✅
```

Once confirmed, the track stays confirmed for its entire lifetime — even if similarity dips temporarily (e.g., the person turns away).

---

## Why OSNet and Not Other Models?

| Model | Parameters | Speed | Accuracy | Why/Why Not |
|-------|-----------|-------|----------|-------------|
| **OSNet x1_0** ✅ | 2.2M | Fast | High | Best accuracy-to-speed ratio; captures multi-scale features; lightweight enough for real-time |
| ResNet-50 | 25.6M | Slow | Good | Too heavy for per-frame inference on every detected person |
| MobileNet-v2 | 3.4M | Very Fast | Lower | Not designed for ReID; misses fine-grained person details |
| BoT (Bag of Tricks) | 25.6M | Slow | Very High | Overkill for single-video search; too slow per crop |
| OSNet x0.25 | 0.2M | Fastest | Moderate | Too small; loses fine details needed for reliable matching |

OSNet x1_0 hits the sweet spot: **accurate enough to distinguish similar-looking people, fast enough to run on every detection in every frame**.

---

## The Role of SAM in Improving OSNet

Without SAM:
```
┌─────────────────────┐
│ ██████Person████████ │  ← YOLO crop includes background corners
│ █Background█Person█ │     between arms, legs, and around head
│ ██████Person████████ │
└─────────────────────┘
         │
    OSNet embeds background pixels too → noisy embedding
```

With SAM:
```
┌─────────────────────┐
│ ▒▒▒▒▒▒Person▒▒▒▒▒▒▒ │  ← SAM masks background to ImageNet mean
│ ▒ImageNet▒▒Person▒▒ │     (neutral grey that OSNet ignores)
│ ▒▒mean▒▒▒Person▒▒▒▒ │
└─────────────────────┘
         │
    OSNet embeds only person pixels → cleaner embedding
```

The ImageNet mean fill `(104, 116, 124)` in BGR is the value that, after standardisation, becomes `(0, 0, 0)` in the normalised input — essentially "zero signal" that doesn't activate any features in the network. This means **only the person's pixels contribute to the embedding**.

---

## Embedding Space Visualisation (Conceptual)

Imagine the 512-dimensional space compressed to 2D:

```
                    Embedding Space
                    
        ●A₁  ●A₂                    ● = reference embedding
       ●A₃                           ○ = video crop embedding
         ●A₄
                                      A = Person A (target POI)
                                      B = Person B (bystander)
   ○ₐ₁  ○ₐ₂                         C = Person C (bystander)
      ○ₐ₃         similarity
                   threshold
                   ─ ─ ─ ─ ─
                                  ○c₁  ○c₂
            ○b₁                    ●C₁
               ○b₂  ●B₁
                       ●B₂
```

- The reference embeddings for Person A (●A₁–A₄) cluster together.
- Video crops of the same person (○a₁–a₃) land near them → high similarity → match.
- Video crops of different people (○b, ○c) land far away → low similarity → no match.

This clustering property is what OSNet was trained to achieve: **minimise intra-class distance** (same person, different images) and **maximise inter-class distance** (different people).

---

## Summary

| Step | What Happens | Component |
|------|-------------|-----------|
| 1 | Load pretrained OSNet x1_0 | `torchreid` |
| 2 | Resize reference crop to 256×128, ImageNet normalise | `_preprocess()` |
| 3 | Forward pass through OSNet → 512-d vector, L2 normalised | `extract_embedding()` |
| 4 | Augment reference (5 variants), average embeddings | `_augment()` + `encode_references()` |
| 5 | For each video frame crop: embed → cosine similarity vs gold | `match_any()` |
| 6 | Majority vote: 6/10 frames match → confirmed POI | `process_image_mode()` |

OSNet is the brain that answers one question: **"Is this the same person?"** — and it answers it 512 numbers at a time, hundreds of times per second.
