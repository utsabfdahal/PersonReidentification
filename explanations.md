# Person of Interest (POI) Tracker & Extractor — Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Configuration — config.yaml](#4-configuration--configyaml)
5. [Core Modules](#5-core-modules)
   - 5.1 [Detector — src/detector.py](#51-detector--srcdetectorpy)
   - 5.2 [ReID Engine — src/reid_engine.py](#52-reid-engine--srcreid_enginepy)
   - 5.3 [SAM Segmentor — src/segmentor.py](#53-sam-segmentor--srcsegmentorpy)
   - 5.4 [Video Tools — src/video_tools.py](#54-video-tools--srcvideo_toolspy)
6. [CLI Entry Point — main.py](#6-cli-entry-point--mainpy)
7. [Web Frontend — app.py & index.html](#7-web-frontend--apppy--indexhtml)
8. [Processing Pipelines](#8-processing-pipelines)
   - 8.1 [Image Mode (Engine A): Visual Re-Identification](#81-image-mode-engine-a-visual-re-identification)
   - 8.2 [Text Mode (Engine B): Semantic Grounding](#82-text-mode-engine-b-semantic-grounding)
9. [AI Models Used](#9-ai-models-used)
10. [Key Algorithms & Techniques](#10-key-algorithms--techniques)
11. [Installation & Setup](#11-installation--setup)
12. [Usage](#12-usage)
13. [How a Frame Is Processed (Step-by-Step)](#13-how-a-frame-is-processed-step-by-step)
14. [Output & Video Export](#14-output--video-export)
15. [Technology Stack](#15-technology-stack)

---

## 1. Project Overview

The **POI Tracker & Extractor** is a video analysis system that automatically locates and extracts segments of a Person of Interest (POI) from surveillance or general video footage. It supports two detection modes:

| Mode | How You Identify the POI | Model Stack |
|------|--------------------------|-------------|
| **Image Mode (Engine A)** | Upload one or more reference photos of the target person | YOLO11n + OSNet (Re-ID) + MobileSAM |
| **Text Mode (Engine B)** | Type a natural-language description (e.g., *"person in a red shirt"*) | YOLO-World |

The system processes every frame of the input video, tracks detected persons across frames using ByteTrack, confirms POI identity through a majority-voting mechanism, and exports a new video clip containing only the segments where the POI appears — with bounding boxes, labels, and optional SAM segmentation masks overlaid.

---

## 2. System Architecture

```
                         ┌─────────────────────┐
                         │     User Input       │
                         │  (CLI or Web UI)     │
                         └─────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              Image Mode                    Text Mode
              (--image)                     (--text)
                    │                             │
         ┌──────────┴──────────┐                  │
         │                     │                  │
    YOLO11n (detect)    MobileSAM (segment)  YOLO-World
         │                     │             (detect+track)
         │              ┌──────┘                  │
         │              │                         │
    ┌────┴────┐    Mask background                │
    │ Person  │    of reference crops              │
    │ Crops   │────────┐                          │
    └─────────┘        │                          │
                       ▼                          │
                OSNet (ReID)                      │
              Extract 512-dim                     │
              embeddings                          │
                       │                          │
         ┌─────────────┴──────────────────────────┘
         │
         ▼
    ByteTrack (multi-object tracking)
         │
         ▼
    Majority Voting (confirm track IDs)
         │
         ▼
    Annotated Frames (bbox / SAM mask overlay)
         │
         ▼
    OpenCV → ffmpeg H.264 re-encode
         │
         ▼
    Output POI Video Clip
```

---

## 3. Directory Structure

```
AmritProject/
├── main.py                  # CLI entry point (argparse-based)
├── app.py                   # Flask web server (port 5050)
├── config.yaml              # All tuneable parameters
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
│
├── src/                     # Core library modules
│   ├── __init__.py          # Package exports
│   ├── detector.py          # MultiModalDetector (YOLO / YOLO-World)
│   ├── reid_engine.py       # ReIDEngine (OSNet embeddings & matching)
│   ├── segmentor.py         # SAMSegmentor (MobileSAM masks)
│   └── video_tools.py       # Drawing, segment math, video export
│
├── templates/
│   └── index.html           # Web UI (dark-themed, drag-drop uploads)
│
├── input/                   # Place input video files here
├── ref/                     # Place reference person images here
├── output/                  # Generated POI clips appear here
├── uploads/                 # Web UI temporary uploads (per-job)
├── static/                  # Static web assets (currently empty)
│
├── venv/                    # Python virtual environment
├── yolo11n.pt               # YOLO11n model weights (auto-downloaded)
├── yolov8s-worldv2.pt       # YOLO-World weights (auto-downloaded)
└── mobile_sam.pt            # MobileSAM weights (auto-downloaded)
```

---

## 4. Configuration — config.yaml

All tuneable parameters live in `config.yaml`. The system loads this at startup and fills in sensible defaults for any missing keys.

| Section | Key | Default | Purpose |
|---------|-----|---------|---------|
| **Models** | `yolo_model` | `"yolo11n.pt"` | Standard YOLO for person detection (image mode) |
| | `yolo_world_model` | `"yolov8s-worldv2.pt"` | YOLO-World for open-vocabulary detection (text mode) |
| | `reid_model_name` | `"osnet_x1_0"` | OSNet variant for generating ReID embeddings |
| | `reid_input_size` | `[256, 128]` | Height × Width for OSNet input preprocessing |
| | `sam_model` | `"mobile_sam.pt"` | MobileSAM model for person segmentation |
| | `use_sam` | `true` | Enable/disable SAM for reference masking & visualization |
| | `sam_on_video` | `false` | Apply SAM to every video crop (slower, potentially better ReID) |
| **Detection** | `person_class_id` | `0` | COCO class ID for "person" |
| | `conf_threshold` | `0.30` | YOLO confidence threshold (lower = more recall) |
| | `min_box_size` | `30` | Skip detections smaller than this (in pixels) |
| **ReID** | `similarity_threshold` | `0.70` | Cosine similarity cutoff for positive match |
| | `ref_augment` | `true` | Generate augmented embeddings from reference images |
| | `voting_window` | `10` | Number of recent frames to consider for voting |
| | `voting_ratio` | `0.6` | Fraction of frames that must match to confirm a track as POI |
| **Text Mode** | `text_conf_threshold` | `0.25` | Lower confidence threshold for noisier YOLO-World detections |
| | `text_voting_window` | `8` | Voting window for text mode |
| | `text_voting_ratio` | `0.5` | Voting ratio for text mode |
| **Export** | `buffer_seconds` | `1.0` | Padding (seconds) added before/after POI segments |
| **Drawing** | `bbox_color` | `[0, 255, 0]` | BGR colour for bounding boxes (green) |
| | `bbox_thickness` | `2` | Box/contour line thickness in pixels |
| | `label_font_scale` | `0.6` | Font size for overlay labels |
| **Paths** | `input_video` | `"input/input_video.mp4"` | Default input (overridden by CLI/web) |
| | `ref_dir` | `"ref"` | Default reference image directory |
| | `output_clip` | `"output/poi_only_clip.mp4"` | Default output path |

---

## 5. Core Modules

### 5.1 Detector — `src/detector.py`

**Class: `MultiModalDetector`**

A unified wrapper around YOLO models that hides whether the system is running in image mode or text mode.

| Method | Purpose |
|--------|---------|
| `init_image_mode()` | Loads YOLOv11n for standard person detection. The model is filtered to COCO class 0 ("person") during tracking. |
| `init_text_mode(prompt)` | Loads YOLO-World (yolov8s-worldv2) and sets the custom class vocabulary to the user's text prompt via `set_classes([prompt])`. |
| `detect_persons(image)` | Runs single-image inference to detect all persons. Returns bounding boxes sorted by area (largest first). Used to crop persons from reference photos before embedding. |
| `track(source)` | Streams `model.track()` with ByteTrack (`bytetrack.yaml`) over the input video. Returns a generator yielding per-frame results with tracked bounding boxes and persistent track IDs. |

**Key design choice:** In image mode, `track()` filters detections to `classes=[0]` (persons only). In text mode, YOLO-World's vocabulary is already set to the user's prompt, so every detection is a candidate POI — no class filtering is needed.

---

### 5.2 ReID Engine — `src/reid_engine.py`

**Class: `ReIDEngine`**

Handles all operations related to person Re-Identification using the OSNet deep learning model.

| Method | Purpose |
|--------|---------|
| `load()` | Builds and loads the pretrained OSNet model (default: `osnet_x1_0`, 512-dimensional embeddings). Moves model to the best available device (CUDA > MPS > CPU). |
| `_preprocess(crop_bgr)` | Resizes the crop to 256×128, converts BGR→RGB, normalises with ImageNet mean/std, returns a PyTorch tensor. |
| `extract_embedding(crop_bgr)` | Preprocesses a person crop and runs a forward pass through OSNet. Returns an L2-normalised 512-dim numpy vector. |
| `_augment(img_bgr)` | Generates 5 variants of a reference image: original, horizontal flip, brightness ×0.85, brightness ×1.15, and a centre crop (10% margins). |
| `encode_references(ref_dir, augment, detector, segmentor)` | The most important method. For each image in `ref_dir`: (1) run YOLO person detection to crop the person, (2) optionally apply SAM to mask out the background, (3) generate augmented versions, (4) compute embeddings for all variants, (5) average and L2-normalise to produce a single "gold standard" embedding per reference. |
| `match_any(embedding, gold_embeddings, threshold)` | Computes cosine similarity between a video crop's embedding and all gold embeddings. Returns `(is_match, best_similarity, matched_name)`. |

**Why YOLO-crop references?** If you use the full reference photo (which may include background, furniture, other people), the embedding captures scene context — not just the person. The video crops from YOLO tracking are tight person-only boxes. This domain mismatch causes false matches. By YOLO-cropping the reference images first, both domains match.

**Why SAM-mask references?** Even after YOLO cropping, the rectangular crop still contains background pixels (corners, between limbs). SAM produces a pixel-level person silhouette, and all background pixels are replaced with ImageNet mean values — so the embedding focuses purely on the person's appearance.

---

### 5.3 SAM Segmentor — `src/segmentor.py`

**Class: `SAMSegmentor`**

Wraps Meta's Segment Anything Model (MobileSAM variant) via the Ultralytics library for bounding-box-prompted segmentation.

| Method | Purpose |
|--------|---------|
| `load()` | Loads `mobile_sam.pt` from Ultralytics (auto-downloads on first run, ~39 MB). |
| `segment_box(image, bbox)` | Given a full image and a bounding box `(x1, y1, x2, y2)`, runs SAM prediction with the box as a prompt. Returns a binary mask (`H×W`, `uint8`, values 0 or 255) where 255 = person pixels. Returns `None` if segmentation fails. |
| `masked_crop(image, bbox)` | Crops the image to the bounding box, then uses `segment_box()` to identify person pixels. Background pixels are filled with ImageNet mean BGR `(104, 116, 124)` instead of black (black would bias the neural network embeddings). Falls back to a raw crop if SAM fails. |
| `draw_mask_contour(frame, mask, color, thickness, alpha)` | Static method that draws a semi-transparent coloured overlay on person pixels and a contour outline around the mask boundary. Modifies the frame in-place. |

**Three uses of SAM in the pipeline:**

1. **Reference encoding** — Mask reference crops before computing OSNet embeddings (cleaner gold standards).
2. **Video crops** (optional, off by default for speed) — Mask each video crop before ReID matching.
3. **Visualization** — Draw segmentation mask contours on confirmed POI frames in the output video instead of plain rectangles.

---

### 5.4 Video Tools — `src/video_tools.py`

Utility functions for drawing annotations, computing temporal segments, and exporting the final video.

| Function | Purpose |
|----------|---------|
| `draw_poi_box(frame, x1, y1, x2, y2, label, cfg, mask)` | Draws a labelled bounding box or SAM mask contour on a frame. If a SAM `mask` is provided, it draws the semi-transparent segmentation overlay; otherwise, it draws a plain rectangle. Always draws a filled label background with the POI info text. |
| `frames_to_segments(poi_frames, fps, buffer_sec, total_frames)` | Converts a list of frame indices where the POI was detected into continuous time segments `(start_seconds, end_seconds)`. Applies configurable buffering (e.g., 1 second before/after) and merges overlapping segments. |
| `export_poi_clip(video_path, segments, annotated_frames, output_path)` | Reads the original video, extracts only the POI segments, substitutes annotated frames where available, writes to a temporary `.mp4` via OpenCV (`mp4v` codec), then re-encodes to H.264 using ffmpeg. |
| `_reencode_h264(src, dst)` | Calls `ffmpeg` to re-encode from OpenCV's `mp4v` codec to `libx264`/`yuv420p` with `faststart` flag — necessary because browsers can only play H.264-encoded MP4 files, not raw `mp4v`. Falls back gracefully if ffmpeg is not installed. |

---

## 6. CLI Entry Point — main.py

The command-line interface uses Python's `argparse` module.

### Arguments

| Flag | Short | Required | Description |
|------|-------|----------|-------------|
| `--video` | `-v` | Yes | Path to input video file |
| `--image` | `-i` | One of `--image`/`--text` | Path to reference image file or directory |
| `--text` | `-t` | One of `--image`/`--text` | Text description of the POI |
| `--output` | `-o` | No | Override output clip path |
| `--config` | `-c` | No | Path to custom config YAML (default: `config.yaml`) |

### Flow

1. Parse arguments and load configuration.
2. Detect the best compute device (CUDA → MPS → CPU).
3. Initialize the detector in the appropriate mode.
4. If `use_sam` is enabled, load MobileSAM.
5. **Image mode:** Load OSNet, encode reference images (with YOLO cropping + SAM masking + augmentation), run frame-by-frame tracking with ReID matching.  
   **Text mode:** Run YOLO-World tracking with the text prompt.
6. Apply majority voting to confirm track IDs.
7. Export the POI-only clip.

---

## 7. Web Frontend — app.py & index.html

### Backend (app.py)

A Flask application running on port **5050** with:

- `/` — Serves the single-page web UI.
- `POST /upload` — Accepts form data (video file + reference images or text prompt), creates a unique job ID, and launches the pipeline in a background thread.
- `GET /status/<job_id>` — Returns the current job status as JSON. The frontend polls this every 2 seconds.
- `GET /result/<filename>` — Serves the output video file for streaming/download.

**Job lifecycle:** `queued` → `loading_models` → `encoding_references` (image mode only) → `processing_video` → `exporting_clip` → `done` (or `error`).

Each upload gets an isolated workspace under `uploads/<job_id>/` with the video and reference images. The output clip goes to `output/<job_id>_poi_clip.mp4`.

### Frontend (index.html)

A dark-themed single-page application with:

- **Mode toggle** — Switch between Image (ReID) and Text (YOLO-World) modes.
- **Drag-and-drop upload** — For both video and reference images, with thumbnail previews.
- **Text prompt input** — For text mode, a free-text field for the person description.
- **Progress bar** — Animated progress with status messages polled from the backend.
- **Video player** — Inline `<video>` element for watching the result directly in the browser, plus a download link.

---

## 8. Processing Pipelines

### 8.1 Image Mode (Engine A): Visual Re-Identification

```
Reference Images (ref/)
        │
        ▼
  ┌─────────────────────┐
  │ YOLO11n: detect     │ ← finds the person in each reference photo
  │ largest person crop  │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │ MobileSAM: segment  │ ← removes background from the crop
  │ fill bg w/ ImageNet │
  │ mean BGR            │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │ Augment (5 variants)│ ← original, flip, bright+, bright-, centre crop
  │ OSNet: embed each   │
  │ Average + L2 norm   │
  └─────────┬───────────┘
            │
            ▼
     Gold Embeddings
     (one 512-d vector
      per reference)
            │
            ▼
  ┌─────────────────────────────────────┐
  │ For each video frame:               │
  │  1. YOLO11n track (ByteTrack)       │
  │  2. For each tracked person:        │
  │     a. Crop (optionally SAM-mask)   │
  │     b. OSNet embed                  │
  │     c. Cosine similarity vs gold    │
  │     d. Append vote (match/no-match) │
  │  3. Check voting window (6/10)      │
  │  4. If confirmed: draw annotation   │
  └─────────────────────────────────────┘
            │
            ▼
     POI Frame Indices
            │
            ▼
  ┌─────────────────────┐
  │ Merge into segments │ ← with 1s buffer padding
  │ Export via OpenCV    │
  │ Re-encode H.264     │
  └─────────────────────┘
```

### 8.2 Text Mode (Engine B): Semantic Grounding

```
  Text Prompt: "person in a red shirt"
            │
            ▼
  ┌─────────────────────────┐
  │ YOLO-World: set_classes │ ← vocabulary = [prompt]
  │ Track with ByteTrack    │
  └─────────┬───────────────┘
            │
            ▼
  ┌─────────────────────────────────────┐
  │ For each video frame:               │
  │  1. Every detection = a candidate   │
  │  2. Track ID voting (4/8)           │
  │  3. If confirmed: draw annotation   │
  └─────────────────────────────────────┘
            │
            ▼
     (same segment merging & export)
```

Text mode is simpler because YOLO-World already filters detections to match the text description — there's no separate ReID step. Voting still helps filter out false positives.

---

## 9. AI Models Used

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| **YOLOv11n** (`yolo11n.pt`) | ~6 MB | Person detection & tracking (image mode) | Ultralytics (auto-download) |
| **YOLO-World v2** (`yolov8s-worldv2.pt`) | ~47 MB | Open-vocabulary detection by text prompt | Ultralytics (auto-download) |
| **OSNet x1_0** | ~8 MB | Person Re-Identification (512-dim embeddings) | torchreid (auto-download) |
| **MobileSAM** (`mobile_sam.pt`) | ~39 MB | Pixel-level person segmentation from box prompts | Ultralytics (auto-download) |
| **ByteTrack** | Config only | Multi-object tracking (persistent track IDs across frames) | Built into Ultralytics |

---

## 10. Key Algorithms & Techniques

### Cosine Similarity Matching

The ReID engine compares embeddings using cosine similarity:

$$\text{similarity}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

A similarity above the threshold (default 0.70) counts as a positive match for a given frame.

### Majority Voting

To reduce false positives, a track ID must be matched in at least 60% of the last 10 frames (6 out of 10) before it is "confirmed" as the POI. Once confirmed, the track remains confirmed for its lifetime.

This prevents momentary misidentifications from producing false output.

### Reference Augmentation

Each reference image is augmented into 5 variants before embedding:

1. **Original** — unmodified crop
2. **Horizontal flip** — mirrors the person
3. **Brightness decrease** (×0.85) — simulates shadows
4. **Brightness increase** (×1.15) — simulates overexposure
5. **Centre crop** (10% margin trim) — simulates different framing

All 5 embeddings are averaged and L2-normalised to produce a robust gold-standard vector that is more tolerant of appearance variations.

### SAM Background Masking

SAM produces a pixel-level binary mask for the person. Instead of zeroing out background pixels (which would create artificial dark regions that bias the embedding), the background is filled with the **ImageNet mean** in BGR: `(104, 116, 124)`. This is the value the neural network treats as "neutral" during inference.

### Temporal Segment Merging

POI detections on individual frames are merged into continuous time segments:

1. Sort all POI frame indices.
2. Group consecutive frames into segments (gap > 2× buffer = new segment).
3. Add buffer padding (default 1 second) before and after each segment.
4. Merge any overlapping segments after buffering.

---

## 11. Installation & Setup

### Prerequisites

- **Python 3.10+** (tested on 3.14)
- **ffmpeg** (for H.264 re-encoding; install via `brew install ffmpeg` on macOS)
- macOS with Apple Silicon (MPS) or a CUDA GPU recommended for performance

### Steps

```bash
# Clone the repository
git clone git@github.com:utsabfdahal/PersonReidentification.git
cd PersonReidentification

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Model weights are auto-downloaded on first run
```

### Dependencies (requirements.txt)

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLO11, YOLO-World, MobileSAM, ByteTrack |
| `opencv-python` | Image/video I/O, drawing, preprocessing |
| `torch` | Deep learning runtime |
| `torchvision` | Image transforms |
| `torchreid` | OSNet Re-ID model |
| `moviepy` | Video utilities |
| `numpy` | Numerical operations |
| `gdown` | Google Drive downloads (for pretrained weights) |
| `tensorboard` | Training visualisation (torchreid dependency) |
| `pyyaml` | YAML config parsing |
| `flask` | Web server |

---

## 12. Usage

### CLI — Image Mode

```bash
# Single reference image
python main.py --video input/surveillance.mp4 --image ref/target_person.jpg

# Multiple reference images (directory)
python main.py --video input/surveillance.mp4 --image ref/

# Custom output path
python main.py --video input/surveillance.mp4 --image ref/ -o output/result.mp4
```

### CLI — Text Mode

```bash
python main.py --video input/surveillance.mp4 --text "person in a red shirt with a black backpack"
```

### Web UI

```bash
python app.py
# Open http://localhost:5050 in your browser
```

Then:
1. Select **Image (ReID)** or **Text (YOLO-World)** mode.
2. Drag-and-drop your video file.
3. Upload reference images (image mode) or type a description (text mode).
4. Click **Start Processing** and wait for the progress bar.
5. Watch the result inline or download the POI clip.

---

## 13. How a Frame Is Processed (Step-by-Step)

Here's exactly what happens for a single frame in **image mode**:

1. **YOLO11n** runs person detection on the frame, filtered to class 0 ("person") with confidence ≥ 0.30.

2. **ByteTrack** assigns or maintains a **persistent track ID** for each detected person across frames, handling occlusions and re-entries.

3. For each tracked person bounding box `(x1, y1, x2, y2)`:
   - Skip if the box is smaller than `min_box_size` (30px) on either dimension.
   - **Crop** the person from the frame. If `sam_on_video` is enabled, apply SAM to mask out background pixels in the crop.
   - **Extract embedding** — preprocess the crop (resize to 256×128, ImageNet normalise) and run through OSNet to get a 512-dimensional L2-normalised vector.
   - **Compare** the embedding against all gold-standard reference embeddings using cosine similarity. Record whether the best match exceeds the threshold (0.70).
   - **Append vote** — the match result (True/False) is appended to the track ID's voting history.

4. **Voting check** — for each track ID not yet confirmed, examine the last 10 votes. If ≥ 6 are positive matches, the track is **confirmed** as POI permanently.

5. **Annotation** — if the track is confirmed (or is a current match), draw a labelled bounding box or SAM segmentation mask on the frame. The label shows `POI:<ref_name> ID:<track_id> <similarity_score>`.

6. If any POI was drawn on this frame, save the annotated frame and record the frame index.

7. After all frames are processed, **merge** POI frame indices into time segments and **export** the clip.

---

## 14. Output & Video Export

The export pipeline:

1. **OpenCV** reads the original video and writes only the POI segments to a temporary `.mp4` file using the `mp4v` codec. Where annotated frames exist (with bounding boxes/masks), they replace the raw frames.

2. **ffmpeg** re-encodes the temporary file to H.264 (`libx264`, `yuv420p`, `faststart`), which is required for browser playback. The command:
   ```
   ffmpeg -y -i temp.mp4 -c:v libx264 -preset fast -crf 23
          -pix_fmt yuv420p -movflags +faststart -an output.mp4
   ```

3. If ffmpeg is not available, the system falls back to the raw `mp4v` file (which works in VLC but not in web browsers).

The output video has the same resolution and frame rate as the input, containing only the time segments where the POI was detected, with buffer padding on both sides.

---

## 15. Technology Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.10+ |
| **Detection** | YOLOv11n (Ultralytics) |
| **Open-Vocab Detection** | YOLO-World v2 (Ultralytics) |
| **Tracking** | ByteTrack (Ultralytics built-in) |
| **Re-Identification** | OSNet x1_0 (torchreid) |
| **Segmentation** | MobileSAM (Ultralytics) |
| **Deep Learning Runtime** | PyTorch (CUDA / MPS / CPU) |
| **Computer Vision** | OpenCV |
| **Video Encoding** | ffmpeg (H.264) |
| **Web Backend** | Flask |
| **Web Frontend** | Vanilla HTML/CSS/JavaScript |
| **Configuration** | YAML |
| **Version Control** | Git (GitHub) |
