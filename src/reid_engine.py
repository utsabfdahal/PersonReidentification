"""
reid_engine.py — OSNet-based Re-Identification
================================================
Encapsulates model loading, embedding extraction, reference encoding
(with augmentations), and similarity matching.
"""

import os
import sys
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid

log = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ReIDEngine:
    """Manages the OSNet model and all embedding operations."""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def load(self) -> None:
        model = torchreid.models.build_model(
            name=self.cfg["reid_model_name"],
            num_classes=1000,
            pretrained=True,
        )
        model = model.to(self.device)
        model.eval()
        self.model = model
        log.info("ReID model '%s' loaded.", self.cfg["reid_model_name"])

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        h, w = self.cfg["reid_input_size"]
        img = cv2.resize(crop_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(crop_bgr).to(self.device)
        feat = self.model(tensor)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Reference encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(img_bgr: np.ndarray) -> list[np.ndarray]:
        augments = [img_bgr]
        augments.append(cv2.flip(img_bgr, 1))
        for alpha in (0.85, 1.15):
            augments.append(cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0))
        h, w = img_bgr.shape[:2]
        mh, mw = h // 10, w // 10
        if mh > 0 and mw > 0:
            augments.append(img_bgr[mh:h - mh, mw:w - mw])
        return augments

    def encode_references(self, ref_dir: str, augment: bool = True, detector=None) -> dict[str, np.ndarray]:
        """Encode reference images.  If *detector* is provided, run YOLO
        person detection first and use the largest person crop rather than
        the raw full image.  This is **critical** because OSNet expects
        tight person crops, not arbitrary scene images."""
        if not os.path.isdir(ref_dir):
            log.error("Reference directory not found: %s", ref_dir)
            sys.exit(1)

        embeddings: dict[str, np.ndarray] = {}
        for fname in sorted(os.listdir(ref_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue
            fpath = os.path.join(ref_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                log.warning("Could not read image, skipping: %s", fpath)
                continue

            # --- Crop person using YOLO before embedding ---
            person_img = img
            if detector is not None:
                person_boxes = detector.detect_persons(fpath)
                if person_boxes:
                    x1, y1, x2, y2 = person_boxes[0]  # largest person
                    person_img = img[y1:y2, x1:x2]
                    log.info("Cropped person from '%s': (%d,%d)-(%d,%d) [%dx%d].",
                             fname, x1, y1, x2, y2, x2 - x1, y2 - y1)
                else:
                    log.warning("No person detected in '%s' — using full image.", fname)

            if augment:
                variants = self._augment(person_img)
                embs = [self.extract_embedding(v) for v in variants]
                avg = np.mean(embs, axis=0)
                avg /= np.linalg.norm(avg) + 1e-8
                embeddings[fname] = avg
                log.info("Gold Standard '%s' (avg %d augmentations, dim=%d).",
                         fname, len(variants), avg.shape[0])
            else:
                embeddings[fname] = self.extract_embedding(person_img)
                log.info("Gold Standard '%s' (dim=%d).", fname, embeddings[fname].shape[0])

        if not embeddings:
            log.error("No valid reference images in %s", ref_dir)
            sys.exit(1)

        log.info("Loaded %d reference embedding(s).", len(embeddings))
        return embeddings

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm else 0.0

    def match_any(
        self,
        embedding: np.ndarray,
        gold_embeddings: dict[str, np.ndarray],
        threshold: float,
    ) -> tuple[bool, float, str]:
        best_sim, best_name = -1.0, ""
        for name, gold in gold_embeddings.items():
            sim = self.cosine_similarity(gold, embedding)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_sim > threshold, best_sim, best_name
