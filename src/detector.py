"""
detector.py — MultiModalDetector
=================================
Handles both standard YOLO (for image-mode ReID) and YOLO-World
(for text-mode open-vocabulary detection).  A single class that the
rest of the pipeline calls without caring which engine is active.
"""

import logging
from ultralytics import YOLO

log = logging.getLogger(__name__)


class MultiModalDetector:
    """
    Wraps YOLO / YOLO-World so callers never need to know which model is
    loaded.  Exposes a unified ``track()`` method.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode: str | None = None      # "image" or "text"
        self.model = None
        self.text_prompt: str | None = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def init_image_mode(self) -> None:
        """Load standard YOLOv11 for person detection (image-ReID pipeline)."""
        model_path = self.cfg["yolo_model"]
        log.info("Loading YOLO model (image mode): %s", model_path)
        self.model = YOLO(model_path)
        self.mode = "image"

    def init_text_mode(self, text_prompt: str) -> None:
        """Load YOLO-World and set the custom class to *text_prompt*."""
        model_path = self.cfg["yolo_world_model"]
        log.info("Loading YOLO-World model (text mode): %s", model_path)
        self.model = YOLO(model_path)
        self.model.set_classes([text_prompt])
        self.text_prompt = text_prompt
        self.mode = "text"
        log.info("YOLO-World custom class set to: '%s'", text_prompt)

    # ------------------------------------------------------------------
    # Single-image person detection (for reference cropping)
    # ------------------------------------------------------------------

    def detect_persons(self, image):
        """
        Run person detection on a single image (BGR ndarray or path).
        Returns list of (x1, y1, x2, y2) bounding boxes sorted by area
        (largest first).  Uses the image-mode YOLO model; loads it
        temporarily if not already loaded.
        """
        from ultralytics import YOLO as _YOLO
        model = self.model if self.mode == "image" and self.model else _YOLO(self.cfg["yolo_model"])
        results = model.predict(
            source=image,
            classes=[self.cfg["person_class_id"]],
            conf=self.cfg["conf_threshold"],
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes.xyxy.cpu().numpy()
        # Sort by area descending
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        return [tuple(map(int, boxes[i])) for i in order]

    # ------------------------------------------------------------------
    # Unified tracking
    # ------------------------------------------------------------------

    def track(self, source: str, **kwargs):
        """
        Run ``model.track()`` with settings appropriate for the active mode.
        Returns the generator / list produced by Ultralytics.
        """
        if self.model is None:
            raise RuntimeError("Detector not initialised — call init_image_mode() or init_text_mode() first.")

        conf = self.cfg["conf_threshold"]
        if self.mode == "text":
            conf = self.cfg.get("text_conf_threshold", conf)

        common = dict(
            source=source,
            tracker="bytetrack.yaml",
            conf=conf,
            stream=True,
            persist=True,
            verbose=False,
        )

        if self.mode == "image":
            common["classes"] = [self.cfg["person_class_id"]]

        # Text mode: YOLO-World already has its class set via set_classes();
        # no need to pass classes=[0] — the only class index is 0 (the prompt).

        common.update(kwargs)
        return self.model.track(**common)
