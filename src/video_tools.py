"""
video_tools.py — Drawing, segment math, and video export
==========================================================
"""

import os
import sys
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_poi_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    cfg: dict,
) -> None:
    """Draw a bounding box and label on *frame* in-place."""
    color = cfg["bbox_color"]
    thick = cfg["bbox_thickness"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = cfg["label_font_scale"]
    (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                font, scale, (0, 0, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Segment math
# ---------------------------------------------------------------------------

def frames_to_segments(
    poi_frames: list[int],
    fps: float,
    buffer_sec: float,
    total_frames: int,
) -> list[tuple[float, float]]:
    """Merge frame indices into continuous (start, end) second segments."""
    if not poi_frames:
        return []

    duration = total_frames / fps
    sorted_frames = sorted(set(poi_frames))

    raw: list[tuple[float, float]] = []
    seg_s = sorted_frames[0] / fps
    seg_e = seg_s

    for f in sorted_frames[1:]:
        t = f / fps
        if t - seg_e > 2 * buffer_sec:
            raw.append((seg_s, seg_e))
            seg_s = t
        seg_e = t
    raw.append((seg_s, seg_e))

    buffered = [(max(0.0, s - buffer_sec), min(duration, e + buffer_sec))
                for s, e in raw]

    merged = [buffered[0]]
    for s, e in buffered[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    log.info("Identified %d POI segment(s):", len(merged))
    for i, (s, e) in enumerate(merged):
        log.info("  Segment %d: %.2fs – %.2fs (%.2fs)", i + 1, s, e, e - s)
    return merged


# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------

def export_poi_clip(
    video_path: str,
    segments: list[tuple[float, float]],
    annotated_frames: dict[int, np.ndarray],
    output_path: str,
) -> None:
    """Extract and stitch POI segments, using annotated frames where available."""
    if not segments:
        log.warning("No POI segments to export.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video for export: %s", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for si, (ss, se) in enumerate(segments):
        sf = int(ss * fps)
        ef = int(se * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        for fn in range(sf, ef + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(annotated_frames.get(fn, frame))
        log.info("Wrote segment %d/%d (frames %d–%d).", si + 1, len(segments), sf, ef)

    writer.release()
    cap.release()
    log.info("POI clip saved → %s", output_path)
