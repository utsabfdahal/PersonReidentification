"""
Person Tracker using Ultralytics YOLO + BoT-SORT/ByteTrack
Detects and tracks multiple persons across video frames.
Each person gets a unique ID maintained throughout the video.
"""

import argparse
import cv2
from ultralytics import YOLO


def run_tracker(source, model_name="yolo11n.pt", tracker="botsort.yaml", show=True, save=True):
    """
    Run person tracking on a video source.

    Args:
        source: Video file path, webcam index (0), or RTSP/HTTP stream URL.
        model_name: YOLO model to use (e.g. yolo11n.pt, yolov8n.pt, yolov8s.pt).
        tracker: Tracker config — 'botsort.yaml' or 'bytetrack.yaml'.
        show: Whether to display the video in a window.
        save: Whether to save the output video.
    """
    # Load YOLO model
    model = YOLO(model_name)

    # COCO class 0 = 'person'
    # model.track() runs detection + tracking per frame
    results = model.track(
        source=source,
        classes=[0],          # only detect persons
        tracker=tracker,      # BoT-SORT (default) or ByteTrack
        show=show,            # live display
        save=save,            # save output video
        stream=True,          # stream results frame-by-frame (memory efficient)
        persist=True,         # keep track IDs across frames
        conf=0.4,             # confidence threshold
        iou=0.5,              # IoU threshold for NMS
        verbose=False,
    )

    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            print(f"Frame {frame_idx}: Tracking {len(track_ids)} person(s) — IDs: {track_ids}")
        else:
            print(f"Frame {frame_idx}: No persons detected")

    print("\nTracking complete.")
    if save:
        print(f"Output saved to: runs/detect/")


def run_tracker_custom_draw(source, model_name="yolo11n.pt", tracker="botsort.yaml", output_path="output.mp4"):
    """
    Run person tracking with custom bounding box drawing and ID labels.
    Gives full control over visualization.
    """
    model = YOLO(model_name)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Color palette for different track IDs
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    ]

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking on current frame
        results = model.track(
            source=frame,
            classes=[0],
            tracker=tracker,
            persist=True,
            conf=0.4,
            iou=0.5,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for tid, bbox, conf in zip(track_ids, bboxes, confs):
                x1, y1, x2, y2 = bbox
                color = colors[tid % len(colors)]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = f"Person {tid} ({conf:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            print(f"Frame {frame_idx}: Tracking {len(track_ids)} person(s) — IDs: {track_ids}")

        # Show frame
        cv2.imshow("Person Tracker", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Person Tracker")
    parser.add_argument("source", help="Video file path or webcam index (0)")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model (default: yolo11n.pt)")
    parser.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"],
                        help="Tracker algorithm (default: botsort.yaml)")
    parser.add_argument("--mode", default="builtin", choices=["builtin", "custom"],
                        help="'builtin' uses YOLO's built-in visualization, 'custom' draws manually")
    parser.add_argument("--output", default="output.mp4", help="Output video path (custom mode)")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window")
    parser.add_argument("--no-save", action="store_true", help="Don't save output video")

    args = parser.parse_args()

    # Handle webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    if args.mode == "builtin":
        run_tracker(source, args.model, args.tracker, show=not args.no_show, save=not args.no_save)
    else:
        run_tracker_custom_draw(source, args.model, args.tracker, args.output)
