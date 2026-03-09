"""
POI Tracker & Extractor — Web Frontend (Multi-Modal)
=====================================================
Flask app supporting both image-ReID and text-prompt modes.
"""

import os
import shutil
import uuid
import threading
import logging

from flask import (
    Flask, render_template, request, jsonify, send_from_directory,
)
from werkzeug.utils import secure_filename

from main import load_config, get_device, process_image_mode, process_text_mode
from src.detector import MultiModalDetector
from src.reid_engine import ReIDEngine
from src.video_tools import export_poi_clip

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

UPLOAD_ROOT = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")

_ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job tracking  (job_id → status dict)
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}


def _run_image_pipeline(job_id: str, video_path: str, ref_dir: str, output_path: str):
    """Image-mode pipeline in a background thread."""
    try:
        jobs[job_id]["status"] = "loading_models"
        cfg = load_config()
        cfg["input_video"] = video_path
        cfg["ref_dir"] = ref_dir
        cfg["output_clip"] = output_path
        device = get_device()

        detector = MultiModalDetector(cfg)
        detector.init_image_mode()

        jobs[job_id]["status"] = "encoding_references"
        reid = ReIDEngine(cfg, device)
        reid.load()
        gold = reid.encode_references(ref_dir, augment=cfg.get("ref_augment", True),
                                      detector=detector)

        jobs[job_id]["status"] = "processing_video"
        segments, annotated = process_image_mode(detector, reid, gold, video_path, cfg)

        if segments:
            jobs[job_id]["status"] = "exporting_clip"
            export_poi_clip(video_path, segments, annotated, output_path)
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = os.path.basename(output_path)
        else:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["error"] = "No POI segments found in the video."

    except Exception as exc:
        log.exception("Image pipeline failed for job %s", job_id)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


def _run_text_pipeline(job_id: str, video_path: str, text_prompt: str, output_path: str):
    """Text-mode pipeline in a background thread."""
    try:
        jobs[job_id]["status"] = "loading_models"
        cfg = load_config()
        cfg["input_video"] = video_path
        cfg["output_clip"] = output_path

        detector = MultiModalDetector(cfg)
        detector.init_text_mode(text_prompt)

        jobs[job_id]["status"] = "processing_video"
        segments, annotated = process_text_mode(detector, video_path, cfg)

        if segments:
            jobs[job_id]["status"] = "exporting_clip"
            export_poi_clip(video_path, segments, annotated, output_path)
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = os.path.basename(output_path)
        else:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["error"] = "No POI segments found in the video."

    except Exception as exc:
        log.exception("Text pipeline failed for job %s", job_id)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept video + (reference images OR text prompt), kick off pipeline."""
    video = request.files.get("video")
    mode = request.form.get("mode", "image")
    text_prompt = request.form.get("text_prompt", "").strip()

    if not video or video.filename == "":
        return jsonify({"error": "No video file provided."}), 400

    vid_ext = os.path.splitext(video.filename)[1].lower()
    if vid_ext not in _ALLOWED_VIDEO:
        return jsonify({"error": f"Unsupported video format: {vid_ext}"}), 400

    # Create unique job workspace
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(UPLOAD_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Save video
    vid_name = secure_filename(video.filename)
    video_path = os.path.join(job_dir, vid_name)
    video.save(video_path)

    output_name = f"{job_id}_poi_clip.mp4"
    output_path = os.path.join(OUTPUT_ROOT, output_name)
    jobs[job_id] = {"status": "queued", "result": None, "error": None, "mode": mode}

    if mode == "text":
        if not text_prompt:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": "Text prompt is required for text mode."}), 400

        thread = threading.Thread(
            target=_run_text_pipeline,
            args=(job_id, video_path, text_prompt, output_path),
            daemon=True,
        )
        thread.start()
    else:
        # Image mode — need reference images
        ref_images = request.files.getlist("references")
        if not ref_images or all(f.filename == "" for f in ref_images):
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": "At least one reference image is required for image mode."}), 400

        ref_dir = os.path.join(job_dir, "ref")
        os.makedirs(ref_dir, exist_ok=True)

        saved_refs = 0
        for img in ref_images:
            if img.filename == "":
                continue
            ext = os.path.splitext(img.filename)[1].lower()
            if ext not in _ALLOWED_IMAGE:
                continue
            img_name = secure_filename(img.filename)
            img.save(os.path.join(ref_dir, img_name))
            saved_refs += 1

        if saved_refs == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": "No valid reference images uploaded."}), 400

        thread = threading.Thread(
            target=_run_image_pipeline,
            args=(job_id, video_path, ref_dir, output_path),
            daemon=True,
        )
        thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    """Poll job progress."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job ID."}), 404
    return jsonify(job)


@app.route("/result/<filename>")
def result_file(filename):
    """Serve the output clip."""
    return send_from_directory(OUTPUT_ROOT, filename, mimetype="video/mp4")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
