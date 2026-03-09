"""
segmentor.py — SAM-based Person Segmentation
==============================================
Uses MobileSAM (via Ultralytics) to produce pixel-level masks
from YOLO bounding-box prompts.  Two main uses:

1. **Mask reference & video crops** before feeding to OSNet so that
   background pixels are zeroed out → cleaner ReID embeddings.
2. **Draw segmentation contours** on output frames for better
   visual quality than plain bounding boxes.
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SAMSegmentor:
    """Wraps MobileSAM for bounding-box-prompted segmentation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.model_name = cfg.get("sam_model", "mobile_sam.pt")

    def load(self) -> None:
        from ultralytics import SAM
        log.info("Loading SAM model: %s", self.model_name)
        self.model = SAM(self.model_name)
        log.info("SAM model loaded.")

    # ------------------------------------------------------------------
    # Core segmentation
    # ------------------------------------------------------------------

    def segment_box(self, image, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """
        Given a full image (BGR ndarray) and a bounding box (x1,y1,x2,y2),
        return a binary mask (H×W, uint8, 0/255) for the person inside.
        Returns None if segmentation fails.
        """
        if self.model is None:
            return None

        x1, y1, x2, y2 = bbox
        results = self.model.predict(
            source=image,
            bboxes=[[x1, y1, x2, y2]],
            verbose=False,
        )

        if not results or results[0].masks is None or len(results[0].masks) == 0:
            return None

        # Take the first (best) mask
        mask_tensor = results[0].masks.data[0]  # (H, W) float 0-1
        mask = mask_tensor.cpu().numpy()

        # Resize to original image dimensions if needed
        h, w = image.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return (mask > 0.5).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Crop with background removed
    # ------------------------------------------------------------------

    def masked_crop(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Return the bounding-box crop with background pixels set to the
        ImageNet mean (instead of black, which would bias the embedding).
        Falls back to the raw crop if SAM is unavailable or fails.
        """
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2].copy()

        mask_full = self.segment_box(image, bbox)
        if mask_full is None:
            return crop

        mask_crop = mask_full[y1:y2, x1:x2]

        # Fill background with ImageNet mean (BGR: 124, 116, 104)
        bg = np.array([104, 116, 124], dtype=np.uint8)
        bg_img = np.full_like(crop, bg)
        mask_3c = cv2.merge([mask_crop, mask_crop, mask_crop]) > 0
        result = np.where(mask_3c, crop, bg_img)
        return result

    # ------------------------------------------------------------------
    # Contour drawing
    # ------------------------------------------------------------------

    @staticmethod
    def draw_mask_contour(
        frame: np.ndarray,
        mask: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        alpha: float = 0.25,
    ) -> None:
        """Draw a semi-transparent mask overlay and contour on *frame* in-place."""
        if mask is None:
            return

        binary = (mask > 127).astype(np.uint8)

        # Semi-transparent fill
        overlay = frame.copy()
        overlay[binary == 1] = (
            (1 - alpha) * overlay[binary == 1] + alpha * np.array(color, dtype=np.float64)
        ).astype(np.uint8)
        np.copyto(frame, overlay)

        # Contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, thickness)
