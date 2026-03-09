"""
Person of Interest (POI) Tracker & Extractor — Multi-Modal
============================================================
CLI entry point.

Usage
-----
  # Image mode  – find a person by reference photo(s)
  python main.py --video input/clip.mp4 --image ref/

  # Text mode   – find a person by natural-language description
  python main.py --video input/clip.mp4 --text "person in a red shirt with a black backpack"

  # Override output path
  python main.py --video input/clip.mp4 --image ref/ -o output/result.mp4
"""

import argparse
import os
import sys
import time
import logging
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml

from src.detector import MultiModalDetector
from src.reid_engine import ReIDEngine
from src.video_tools import draw_poi_box, frames_to_segments, export_poi_clip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    if os.path.isfile(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        log.info("Loaded config from %s", path)
    else:
        log.warning("Config file not found (%s) — using defaults.", path)
        cfg = {}

    # Ensure list → tuple for reid_input_size
    if "reid_input_size" in cfg:
        cfg["reid_input_size"] = tuple(cfg["reid_input_size"])
    else:
        cfg["reid_input_size"] = (256, 128)

    # Ensure bbox_color is a tuple
    if "bbox_color" in cfg:
        cfg["bbox_color"] = tuple(cfg["bbox_color"])
    else:
        cfg["bbox_color"] = (0, 255, 0)

    # Fill defaults for anything missing
    defaults = {
        "yolo_model": "yolo11n.pt",
        "yolo_world_model": "yolov8s-worldv2.pt",
        "reid_model_name": "osnet_x1_0",
        "person_class_id": 0,
        "conf_threshold": 0.30,
        "min_box_size": 30,
        "similarity_threshold": 0.75,
        "ref_augment": True,
        "voting_window": 10,
        "voting_ratio": 0.6,
        "text_conf_threshold": 0.25,
        "text_voting_window": 8,
        "text_voting_ratio": 0.5,
        "buffer_seconds": 1.0,
        "bbox_thickness": 2,
        "label_font_scale": 0.6,
        "input_video": "input/input_video.mp4",
        "ref_dir": "ref",
        "output_dir": "output",
        "output_clip": "output/poi_only_clip.mp4",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info("Using device: %s", dev)
    return dev


# ---------------------------------------------------------------------------
# Image-mode processing
# ---------------------------------------------------------------------------

def process_image_mode(
    detector: MultiModalDetector,
    reid: ReIDEngine,
    gold_embeddings: dict[str, np.ndarray],
    video_path: str,
    cfg: dict,
) -> tuple[list[tuple[float, float]], dict[int, np.ndarray]]:
    """Run detection + tracking + ReID (image mode)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log.info("Video: %s | FPS=%.2f | Frames=%d", video_path, fps, total)

    track_votes: dict[int, list[bool]] = defaultdict(list)
    confirmed: set[int] = set()
    poi_frames: list[int] = []
    annotated: dict[int, np.ndarray] = {}

    frame_idx = 0
    for result in detector.track(source=video_path):
        if result.boxes is None or len(result.boxes) == 0 or result.boxes.id is None:
            frame_idx += 1
            continue

        frame_bgr = result.orig_img.copy()
        tids = result.boxes.id.int().cpu().tolist()
        xyxys = result.boxes.xyxy.cpu().numpy()
        poi_here = False

        for tid, xyxy in zip(tids, xyxys):
            x1, y1, x2, y2 = map(int, xyxy)
            if (x2 - x1) < cfg["min_box_size"] or (y2 - y1) < cfg["min_box_size"]:
                continue
            crop = result.orig_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            emb = reid.extract_embedding(crop)
            is_match, sim, matched_name = reid.match_any(emb, gold_embeddings, cfg["similarity_threshold"])
            track_votes[tid].append(is_match)

            if tid not in confirmed:
                w = cfg["voting_window"]
                recent = track_votes[tid][-w:]
                if len(recent) >= w and sum(recent) >= int(w * cfg["voting_ratio"]):
                    confirmed.add(tid)
                    log.info("Track ID %d confirmed as POI (%d/%d).", tid, sum(recent), w)

            if tid in confirmed or is_match:
                poi_here = True
                ref_label = os.path.splitext(matched_name)[0]
                draw_poi_box(frame_bgr, x1, y1, x2, y2,
                             f"POI:{ref_label} ID:{tid} {sim:.2f}", cfg)

        if poi_here:
            poi_frames.append(frame_idx)
            annotated[frame_idx] = frame_bgr

        frame_idx += 1
        if frame_idx % 500 == 0:
            log.info("Processed %d / %d frames …", frame_idx, total)

    log.info("POI detected in %d / %d frames.", len(poi_frames), total)
    segments = frames_to_segments(poi_frames, fps, cfg["buffer_seconds"], total)
    return segments, annotated


# ---------------------------------------------------------------------------
# Text-mode processing
# ---------------------------------------------------------------------------

def process_text_mode(
    detector: MultiModalDetector,
    video_path: str,
    cfg: dict,
) -> tuple[list[tuple[float, float]], dict[int, np.ndarray]]:
    """
    Run YOLO-World tracking (text mode).
    Every detection already IS the POI (the model only looks for the prompt).
    We still use voting on track IDs to reduce false positives.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log.info("Video: %s | FPS=%.2f | Frames=%d", video_path, fps, total)

    track_votes: dict[int, list[bool]] = defaultdict(list)
    confirmed: set[int] = set()
    poi_frames: list[int] = []
    annotated: dict[int, np.ndarray] = {}

    frame_idx = 0
    for result in detector.track(source=video_path):
        if result.boxes is None or len(result.boxes) == 0 or result.boxes.id is None:
            frame_idx += 1
            continue

        frame_bgr = result.orig_img.copy()
        tids = result.boxes.id.int().cpu().tolist()
        xyxys = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        poi_here = False

        for tid, xyxy, conf in zip(tids, xyxys, confs):
            x1, y1, x2, y2 = map(int, xyxy)
            if (x2 - x1) < cfg["min_box_size"] or (y2 - y1) < cfg["min_box_size"]:
                continue

            track_votes[tid].append(True)   # every detection is a candidate

            if tid not in confirmed:
                w = cfg["text_voting_window"]
                recent = track_votes[tid][-w:]
                if len(recent) >= w and sum(recent) >= int(w * cfg["text_voting_ratio"]):
                    confirmed.add(tid)
                    log.info("Track ID %d confirmed as POI via text (%d/%d).", tid, sum(recent), w)

            if tid in confirmed:
                poi_here = True
                prompt_short = (detector.text_prompt or "?")[:30]
                draw_poi_box(frame_bgr, x1, y1, x2, y2,
                             f"POI:\"{prompt_short}\" ID:{tid} {conf:.2f}", cfg)

        if poi_here:
            poi_frames.append(frame_idx)
            annotated[frame_idx] = frame_bgr

        frame_idx += 1
        if frame_idx % 500 == 0:
            log.info("Processed %d / %d frames …", frame_idx, total)

    log.info("POI detected in %d / %d frames.", len(poi_frames), total)
    segments = frames_to_segments(poi_frames, fps, cfg["buffer_seconds"], total)
    return segments, annotated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Modal POI Tracker & Extractor",
    )
    parser.add_argument("--video", "-v", type=str, help="Path to input video")
    parser.add_argument("--image", "-i", type=str, help="Path to reference image or directory (image mode)")
    parser.add_argument("--text", "-t", type=str, help="Text description of POI (text mode)")
    parser.add_argument("--output", "-o", type=str, help="Output clip path")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Config YAML path")
    args = parser.parse_args()

    if not args.image and not args.text:
        parser.error("Provide either --image (for ReID) or --text (for open-vocabulary detection).")

    if args.image and args.text:
        parser.error("Use --image or --text, not both.")

    cfg = load_config(args.config)

    if args.video:
        cfg["input_video"] = args.video
    if args.output:
        cfg["output_clip"] = args.output

    video_path = cfg["input_video"]
    if not os.path.isfile(video_path):
        log.error("Input video not found: %s", video_path)
        sys.exit(1)

    device = get_device()
    detector = MultiModalDetector(cfg)

    start = time.time()

    if args.image:
        # ---- Engine A: Visual ReID ----
        ref_dir = args.image
        if os.path.isfile(ref_dir):
            # single file → treat parent dir as ref dir (convenience)
            ref_dir = os.path.dirname(ref_dir) or "."
        cfg["ref_dir"] = ref_dir

        detector.init_image_mode()
        reid = ReIDEngine(cfg, device)
        reid.load()

        gold = reid.encode_references(ref_dir, augment=cfg.get("ref_augment", True), detector=detector)
        segments, annotated = process_image_mode(detector, reid, gold, video_path, cfg)

    else:
        # ---- Engine B: Semantic Grounding (Text) ----
        detector.init_text_mode(args.text)
        segments, annotated = process_text_mode(detector, video_path, cfg)

    elapsed = time.time() - start
    log.info("Processing took %.1f seconds.", elapsed)

    if segments:
        export_poi_clip(video_path, segments, annotated, cfg["output_clip"])
    else:
        log.warning("No POI segments found — no clip exported.")


if __name__ == "__main__":
    main()
