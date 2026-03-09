"""
Person Identifier using YOLO Tracking + OSNet ReID
Identifies a specific person in CCTV video using a reference image.

Strategy:
1. YOLO detects & tracks all persons (each gets a track ID)
2. For each track ID, we collect person crops from random frames
3. OSNet extracts 512-dim whole-body feature embeddings
4. We compare each crop's embedding against the reference image embedding
5. Majority voting across sampled frames → identify the target person

OSNet is specifically trained for person re-identification:
- Matches based on clothing, body shape, colors, texture
- Robust to viewpoint/angle changes
- Works without needing face visibility (perfect for CCTV)
"""

import argparse
import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import torchreid


# ── ReID Feature Extractor ──────────────────────────────────────────────────

class PersonReID:
    """Whole-body person re-identification using OSNet."""

    def __init__(self, model_name="osnet_x1_0", device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = torch.device(device)

        # Load OSNet pretrained on ImageNet (general features work well for same-outfit matching)
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard ReID preprocessing: resize to 256x128 (height x width)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image_bgr):
        """Extract 512-dim feature embedding from a BGR person crop."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu()

    @torch.no_grad()
    def extract_batch(self, images_bgr):
        """Extract features from a batch of BGR crops."""
        if not images_bgr:
            return torch.empty(0, 512)
        tensors = []
        for img in images_bgr:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu()

    @staticmethod
    def cosine_similarity(feat1, feat2):
        """Compute cosine similarity between two feature vectors."""
        return torch.mm(feat1, feat2.t()).item()


# ── Main Identifier ─────────────────────────────────────────────────────────

def identify_person(
    video_source,
    reference_image_path,
    yolo_model="yolo11n.pt",
    tracker="botsort.yaml",
    sample_frames=5,
    similarity_threshold=0.55,
    output_path="identified_output.mp4",
    show=True,
):
    """
    Two-pass approach:
      Pass 1: Run YOLO tracking, collect crops for each track ID at random frames.
      Pass 2: Match crops against reference → identify target → re-render video with labels.
    """

    # ── Load models ──────────────────────────────────────────────────────
    print("[1/4] Loading models...")
    yolo = YOLO(yolo_model)
    reid = PersonReID()

    # ── Extract reference embedding ──────────────────────────────────────
    print("[2/4] Processing reference image...")
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        print(f"Error: Cannot read reference image: {reference_image_path}")
        return
    ref_embedding = reid.extract(ref_img)
    print(f"  Reference embedding shape: {ref_embedding.shape}")

    # ── Pass 1: Track all persons and collect crops ──────────────────────
    print("[3/4] Pass 1 — Tracking all persons and collecting samples...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # {track_id: [(frame_idx, bbox), ...]}  — all appearances
    track_appearances = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.track(
            source=frame,
            classes=[0],
            tracker=tracker,
            persist=True,
            conf=0.4,
            iou=0.5,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.int().cpu().tolist()

            for tid, bbox in zip(track_ids, bboxes):
                if tid not in track_appearances:
                    track_appearances[tid] = []
                track_appearances[tid].append((frame_idx, bbox))

        frame_idx += 1

    cap.release()

    if not track_appearances:
        print("No persons detected in the video.")
        return

    print(f"  Found {len(track_appearances)} unique track(s): {list(track_appearances.keys())}")
    for tid, appearances in track_appearances.items():
        print(f"    Track {tid}: {len(appearances)} frames")

    # ── Sample random frames and extract ReID features ───────────────────
    print("  Sampling frames and computing ReID features...")

    # For each track, pick `sample_frames` random appearances
    target_track_ids = set()
    track_similarities = {}

    cap = cv2.VideoCapture(video_source)

    for tid, appearances in track_appearances.items():
        # Sample random frames (or all if fewer than sample_frames)
        n_samples = min(sample_frames, len(appearances))
        sampled = random.sample(appearances, n_samples)

        crops = []
        for (fidx, bbox) in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            x1, y1, x2, y2 = bbox
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        if not crops:
            continue

        # Extract features for all crops in batch
        crop_embeddings = reid.extract_batch(crops)

        # Compare each crop with reference
        similarities = []
        for i in range(crop_embeddings.shape[0]):
            sim = reid.cosine_similarity(crop_embeddings[i:i+1], ref_embedding)
            similarities.append(sim)

        avg_sim = np.mean(similarities)
        max_sim = np.max(similarities)
        matches = sum(1 for s in similarities if s >= similarity_threshold)

        track_similarities[tid] = {
            "avg": avg_sim,
            "max": max_sim,
            "matches": matches,
            "total": len(similarities),
            "all": similarities,
        }

        print(f"    Track {tid}: avg_sim={avg_sim:.3f}, max_sim={max_sim:.3f}, "
              f"matches={matches}/{len(similarities)}")

        # Accept ALL tracks where majority of sampled frames match
        if matches >= n_samples / 2:
            target_track_ids.add(tid)

    cap.release()

    if not target_track_ids:
        print("\n  ⚠ No matching person found above threshold.")
        print(f"  Try lowering --threshold (current: {similarity_threshold})")
        print("  Similarity scores:")
        for tid, info in sorted(track_similarities.items(), key=lambda x: -x[1]["avg"]):
            print(f"    Track {tid}: avg={info['avg']:.3f}, max={info['max']:.3f}")
        return

    print(f"\n  ✓ TARGET IDENTIFIED: Track IDs {sorted(target_track_ids)}")
    for tid in sorted(target_track_ids):
        info = track_similarities[tid]
        print(f"    Track {tid}: avg_sim={info['avg']:.3f}, max_sim={info['max']:.3f}")

    # ── Pass 2: Re-render video with identification + all-tracked labels ─
    print(f"[4/4] Pass 2 — Rendering output videos...")

    # Derive all-tracked video path from output_path
    base, ext = os.path.splitext(output_path)
    all_tracked_path = f"{base}_all_tracked{ext}"

    cap = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_id = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    out_all = cv2.VideoWriter(all_tracked_path, fourcc, fps, (w, h))

    # Color palette for all-tracked video
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    ]

    # Reset YOLO tracker state for fresh pass
    yolo2 = YOLO(yolo_model)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo2.track(
            source=frame,
            classes=[0],
            tracker=tracker,
            persist=True,
            conf=0.4,
            iou=0.5,
            verbose=False,
        )

        frame_id = frame.copy()
        frame_all = frame.copy()
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for tid, bbox, conf in zip(track_ids, bboxes, confs):
                x1, y1, x2, y2 = bbox

                # ── Identified video (target = green, others = gray) ──
                is_target = (tid in target_track_ids)
                if is_target:
                    color = (0, 255, 0)
                    thickness = 3
                    label = f"TARGET ({conf:.2f})"
                else:
                    color = (128, 128, 128)
                    thickness = 1
                    label = f"Person {tid}"

                cv2.rectangle(frame_id, (x1, y1), (x2, y2), color, thickness)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame_id, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame_id, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # ── All-tracked video (unique color per track ID) ──
                t_color = palette[tid % len(palette)]
                cv2.rectangle(frame_all, (x1, y1), (x2, y2), t_color, 2)
                t_label = f"Person {tid} ({conf:.2f})"
                (tw2, th2), _ = cv2.getTextSize(t_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_all, (x1, y1 - th2 - 10), (x1 + tw2 + 4, y1), t_color, -1)
                cv2.putText(frame_all, t_label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show:
            cv2.imshow("Identified", frame_id)
            cv2.imshow("All Tracked", frame_all)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out_id.write(frame_id)
        out_all.write(frame_all)
        frame_idx += 1

    cap.release()
    out_id.release()
    out_all.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Output saved:")
    print(f"  Identified  → {output_path}")
    print(f"  All tracked → {all_tracked_path}")
    print(f"Target person (Tracks {sorted(target_track_ids)}) highlighted in GREEN.")


# ── Single-Pass Real-Time Mode ──────────────────────────────────────────────

def identify_person_realtime(
    video_source,
    reference_image_path,
    yolo_model="yolo11n.pt",
    tracker="botsort.yaml",
    check_interval=30,
    sample_frames=5,
    similarity_threshold=0.55,
    output_path="identified_realtime.mp4",
    show=True,
):
    """
    Single-pass real-time mode:
    - Tracks persons continuously
    - Periodically checks each new track ID against the reference
    - Once identified, labels the target person in real-time
    """

    print("[1/3] Loading models...")
    yolo = YOLO(yolo_model)
    reid = PersonReID()

    print("[2/3] Processing reference image...")
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        print(f"Error: Cannot read reference image: {reference_image_path}")
        return
    ref_embedding = reid.extract(ref_img)

    print("[3/3] Running real-time identification...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # State for each track: collected crops and identification status
    # {track_id: {"crops": [crop_img, ...], "identified": bool, "is_target": bool, "sim": float}}
    track_state = {}
    identified_targets = set()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.track(
            source=frame,
            classes=[0],
            tracker=tracker,
            persist=True,
            conf=0.4,
            iou=0.5,
            verbose=False,
        )

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for tid, bbox, conf in zip(track_ids, bboxes, confs):
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Initialize state for new tracks
                if tid not in track_state:
                    track_state[tid] = {
                        "crops": [],
                        "frame_indices": [],
                        "identified": False,
                        "is_target": False,
                        "sim": 0.0,
                    }

                state = track_state[tid]

                # Collect crops at intervals until we have enough samples
                if (not state["identified"]
                        and len(state["crops"]) < sample_frames
                        and frame_idx % check_interval == 0):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        state["crops"].append(crop.copy())
                        state["frame_indices"].append(frame_idx)

                # Once we have enough crops, run identification
                if not state["identified"] and len(state["crops"]) >= sample_frames:
                    crop_embeddings = reid.extract_batch(state["crops"])
                    similarities = []
                    for i in range(crop_embeddings.shape[0]):
                        sim = reid.cosine_similarity(crop_embeddings[i:i+1], ref_embedding)
                        similarities.append(sim)

                    avg_sim = np.mean(similarities)
                    matches = sum(1 for s in similarities if s >= similarity_threshold)

                    state["identified"] = True
                    state["sim"] = avg_sim

                    if matches >= sample_frames / 2:
                        state["is_target"] = True
                        identified_targets.add(tid)
                        print(f"  ✓ Frame {frame_idx}: Track {tid} IDENTIFIED as target "
                              f"(avg_sim={avg_sim:.3f}, matches={matches}/{len(similarities)})")
                    else:
                        print(f"    Frame {frame_idx}: Track {tid} is NOT target "
                              f"(avg_sim={avg_sim:.3f}, matches={matches}/{len(similarities)})")

                    # Free memory
                    state["crops"] = []

                # Draw
                is_target = state["is_target"]
                if is_target:
                    color = (0, 255, 0)
                    thickness = 3
                    label = f"TARGET ({conf:.2f})"
                elif not state["identified"]:
                    color = (0, 165, 255)   # ORANGE = pending identification
                    thickness = 2
                    label = f"Checking... [{len(state['crops'])}/{sample_frames}]"
                else:
                    color = (128, 128, 128)
                    thickness = 1
                    label = f"Person {tid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show:
            cv2.imshow("Person Identifier (Real-time)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Processed {frame_idx} frames.")
    print(f"Target person(s) found: {identified_targets or 'None'}")
    print(f"Output saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify a specific person in CCTV video using a reference image (whole-body ReID)"
    )
    parser.add_argument("source", help="Video file path or webcam index (0)")
    parser.add_argument("--ref", required=True, help="Path to reference image of the target person")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model (default: yolo11n.pt)")
    parser.add_argument("--tracker", default="botsort.yaml",
                        choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--mode", default="twopass", choices=["twopass", "realtime"],
                        help="'twopass' = more accurate (2 passes), 'realtime' = single pass")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of frames to sample per person for matching (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Cosine similarity threshold for match (default: 0.55)")
    parser.add_argument("--check-interval", type=int, default=30,
                        help="Collect a crop every N frames in realtime mode (default: 30)")
    parser.add_argument("--output", default="identified_output.mp4", help="Output video path")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window")

    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    if args.mode == "twopass":
        identify_person(
            video_source=source,
            reference_image_path=args.ref,
            yolo_model=args.model,
            tracker=args.tracker,
            sample_frames=args.samples,
            similarity_threshold=args.threshold,
            output_path=args.output,
            show=not args.no_show,
        )
    else:
        identify_person_realtime(
            video_source=source,
            reference_image_path=args.ref,
            yolo_model=args.model,
            tracker=args.tracker,
            check_interval=args.check_interval,
            sample_frames=args.samples,
            similarity_threshold=args.threshold,
            output_path=args.output,
            show=not args.no_show,
        )
