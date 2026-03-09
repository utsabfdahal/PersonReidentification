"""Quick demo: YOLO + ByteTrack output showing all tracked persons."""
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import os

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("input/input_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output/bytetrack_demo.tmp.mp4", fourcc, fps, (w, h))

colors = {}

def get_color(tid):
    if tid not in colors:
        rng = np.random.RandomState(tid * 37)
        colors[tid] = tuple(int(c) for c in rng.randint(80, 255, 3))
    return colors[tid]

frame_idx = 0
track_summary = {}

for result in model.track(
    source="input/input_video.mp4",
    tracker="bytetrack.yaml",
    conf=0.30,
    classes=[0],
    stream=True,
    persist=True,
    verbose=False,
):
    frame = result.orig_img.copy()

    if result.boxes is not None and len(result.boxes) > 0 and result.boxes.id is not None:
        tids = result.boxes.id.int().cpu().tolist()
        xyxys = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for tid, xyxy, conf in zip(tids, xyxys, confs):
            x1, y1, x2, y2 = map(int, xyxy)
            color = get_color(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tid} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - baseline - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            if tid not in track_summary:
                track_summary[tid] = {"first": frame_idx, "last": frame_idx, "count": 0}
            track_summary[tid]["last"] = frame_idx
            track_summary[tid]["count"] += 1

    cv2.putText(frame, f"Frame {frame_idx}/{total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    writer.write(frame)
    frame_idx += 1

writer.release()

# Re-encode to H.264
try:
    subprocess.run(
        ["ffmpeg", "-y", "-i", "output/bytetrack_demo.tmp.mp4",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
         "output/bytetrack_demo.mp4"],
        check=True, capture_output=True,
    )
    os.remove("output/bytetrack_demo.tmp.mp4")
except Exception:
    os.rename("output/bytetrack_demo.tmp.mp4", "output/bytetrack_demo.mp4")

print(f"\nProcessed {frame_idx} frames")
print(f"Unique tracks found: {len(track_summary)}\n")
print(f"{'ID':<6} {'Frames':<10} {'First':<8} {'Last':<8} {'Duration':<10}")
print("-" * 42)
for tid in sorted(track_summary):
    s = track_summary[tid]
    dur = (s["last"] - s["first"]) / fps
    print(f"{tid:<6} {s['count']:<10} {s['first']:<8} {s['last']:<8} {dur:.1f}s")

print(f"\nOutput saved: output/bytetrack_demo.mp4")
