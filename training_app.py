#!/usr/bin/env python3
"""Browser UI to manage training classes, images, and run training."""

from __future__ import annotations

import argparse
import mimetypes
import re
import shutil
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from threading import Timer

import cv2
from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from training_data import (
    PROJECT_ROOT,
    copy_first_training_as_display,
    create_class,
    delete_class,
    delete_image,
    import_images,
    list_class_slugs,
    list_images,
    sanitize_slug,
    set_display_image,
    training_root,
)
from training_fit import run_training

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
)
app.secret_key = __import__("secrets").token_hex(16)


def _slug_ok(slug: str) -> bool:
    return bool(re.match(r"^[a-z0-9_\-]+$", slug)) and not slug.startswith("_")


@app.route("/")
def index():
    root = PROJECT_ROOT
    slugs = list_class_slugs(root)
    counts = {s: len(list_images(s, root)) for s in slugs}
    return render_template("index.html", slugs=slugs, counts=counts)


@app.post("/class/add")
def class_add():
    name = request.form.get("name", "").strip()
    try:
        slug = sanitize_slug(name)
        if slug in list_class_slugs(PROJECT_ROOT):
            flash("That class already exists.", "error")
        else:
            create_class(slug, PROJECT_ROOT)
            flash(f"Created class “{slug}”.", "ok")
    except ValueError as e:
        flash(str(e), "error")
    return redirect(url_for("index"))


@app.route("/class/<slug>")
def class_detail(slug: str):
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        flash("Unknown class.", "error")
        return redirect(url_for("index"))
    root = PROJECT_ROOT
    images = list_images(slug, root)
    return render_template("class_detail.html", slug=slug, images=images)


@app.get("/class/<slug>/image/<filename>")
def training_image(slug: str, filename: str):
    """Serve a training image for thumbnails and preview (must exist for this class)."""
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        abort(404)
    safe = secure_filename(filename)
    if not safe:
        abort(404)
    known = {p.name for p in list_images(slug, PROJECT_ROOT)}
    if safe not in known:
        abort(404)
    directory = training_root(PROJECT_ROOT) / slug
    mime, _ = mimetypes.guess_type(safe)
    return send_from_directory(
        directory,
        safe,
        mimetype=mime or "application/octet-stream",
    )


@app.post("/class/<slug>/upload")
def class_upload(slug: str):
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        flash("Unknown class.", "error")
        return redirect(url_for("index"))
    files = request.files.getlist("files")
    paths: list[Path] = []
    tmpdir = Path(tempfile.mkdtemp())
    try:
        for f in files:
            if not f or not f.filename:
                continue
            safe = secure_filename(f.filename)
            if not safe:
                continue
            p = tmpdir / safe
            f.save(p)
            paths.append(p)
        if paths:
            n = len(import_images(slug, paths, PROJECT_ROOT))
            flash(f"Imported {n} image(s).", "ok")
        else:
            flash("No files selected.", "error")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return redirect(url_for("class_detail", slug=slug))


@app.post("/class/<slug>/delete")
def class_delete(slug: str):
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        flash("Unknown class.", "error")
        return redirect(url_for("index"))
    delete_class(slug, PROJECT_ROOT)
    flash(f"Removed class “{slug}”.", "ok")
    return redirect(url_for("index"))


@app.post("/class/<slug>/image/delete")
def image_delete(slug: str):
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        flash("Unknown class.", "error")
        return redirect(url_for("index"))
    name = request.form.get("filename", "")
    safe = secure_filename(name)
    if not safe:
        flash("Invalid filename.", "error")
        return redirect(url_for("class_detail", slug=slug))
    target = training_root(PROJECT_ROOT) / slug / safe
    try:
        delete_image(target, PROJECT_ROOT)
        flash("Image removed.", "ok")
    except ValueError:
        flash("Invalid path.", "error")
    return redirect(url_for("class_detail", slug=slug))


@app.post("/class/<slug>/display_first")
def display_first(slug: str):
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        flash("Unknown class.", "error")
        return redirect(url_for("index"))
    p = copy_first_training_as_display(slug, PROJECT_ROOT)
    if p:
        flash("Display image set from first training photo.", "ok")
    else:
        flash("No training images to copy.", "error")
    return redirect(url_for("class_detail", slug=slug))


@app.post("/class/<slug>/display_upload")
def display_upload(slug: str):
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        flash("Unknown class.", "error")
        return redirect(url_for("index"))
    f = request.files.get("file")
    if not f or not f.filename:
        flash("No file.", "error")
        return redirect(url_for("class_detail", slug=slug))
    safe = secure_filename(f.filename)
    tmpdir = Path(tempfile.mkdtemp())
    tmp = tmpdir / safe
    try:
        f.save(tmp)
        set_display_image(slug, tmp, PROJECT_ROOT)
        flash("Display image updated.", "ok")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return redirect(url_for("class_detail", slug=slug))


@app.post("/train")
def train():
    ok, msg = run_training(PROJECT_ROOT)
    if ok:
        flash(msg, "ok")
    else:
        flash(msg, "error")
    return redirect(url_for("index"))


class CameraThread:
    """Background thread that continuously reads frames from the NoIR camera via picamera2."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._started = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            from picamera2 import Picamera2
            pc2 = Picamera2()
            config = pc2.create_video_configuration(
                main={"size": (640, 480), "format": "BGR888"}
            )
            pc2.configure(config)
            pc2.start()
            try:
                while self._running:
                    frame = pc2.capture_array("main")
                    with self._lock:
                        self._frame = frame.copy()
            finally:
                pc2.stop()
                pc2.close()
        except Exception as exc:
            self.error = str(exc)
            self._running = False

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


_camera = CameraThread()


@app.get("/camera/stream")
def camera_stream():
    """MJPEG stream from the NoIR camera."""
    _camera.ensure_started()

    def generate():
        while True:
            try:
                frame = _camera.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n"
                    )
                time.sleep(0.08)  # ~12 fps
            except GeneratorExit:
                break

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/class/<slug>/capture")
def camera_capture(slug: str):
    """Capture a single frame from the NoIR camera and save as a training image."""
    if not _slug_ok(slug) or slug not in list_class_slugs(PROJECT_ROOT):
        flash("Unknown class.", "error")
        return redirect(url_for("index"))
    _camera.ensure_started()
    frame = _camera.get_frame()
    if frame is None:
        flash("Camera not ready — click Start preview first.", "error")
        return redirect(url_for("class_detail", slug=slug))
    ret, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ret:
        flash("Failed to encode frame.", "error")
        return redirect(url_for("class_detail", slug=slug))
    tmpdir = Path(tempfile.mkdtemp())
    tmp = tmpdir / "capture.jpg"
    try:
        tmp.write_bytes(bytes(jpeg_buf))
        n = len(import_images(slug, [tmp], PROJECT_ROOT))
        flash(f"Saved {n} frame(s) to training data.", "ok")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return redirect(url_for("class_detail", slug=slug))


def main() -> None:
    parser = argparse.ArgumentParser(description="Training manager (browser UI)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    training_root(PROJECT_ROOT).mkdir(parents=True, exist_ok=True)

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"

        def _open() -> None:
            webbrowser.open(url)

        Timer(0.7, _open).start()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
