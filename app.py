#!/usr/bin/env python3
"""
Live hand-pose viewer: camera + landmark overlay + panel image from trained classes.
Train classes and images in the browser: python training_app.py
Then: python train.py   (or Train from the web UI)
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import learned_classifier
from landmark_extract import create_extractor
from training_data import display_root, list_class_slugs, list_images

PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "Chittapur hand pose"


def load_display_images(project_root: Path) -> Dict[str, np.ndarray]:
    """Panel images: data/display/<slug>.png, else first training image per class."""
    learned_classifier.load_model(project_root)
    labels = learned_classifier.known_labels()
    if not labels:
        labels = list_class_slugs(project_root)
    out: Dict[str, np.ndarray] = {}
    for slug in labels:
        disp = display_root(project_root) / f"{slug}.png"
        im = cv2.imread(str(disp)) if disp.is_file() else None
        if im is None:
            imgs = list_images(slug, project_root)
            if imgs:
                im = cv2.imread(str(imgs[0]))
        if im is not None:
            out[slug] = im
    return out


def resize_fit(im: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    h, w = im.shape[:2]
    scale = min(box_w / w, box_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    canvas[:] = (32, 32, 32)
    y0 = (box_h - nh) // 2
    x0 = (box_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def video_capture_open(index: int):
    if platform.system() == "Darwin":
        backend = getattr(cv2, "CAP_AVFOUNDATION", None)
        if backend is not None:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
    elif platform.system() == "Linux":
        try:
            from picam_capture import Picamera2Capture, _num_picamera2_cameras
            if index < _num_picamera2_cameras():
                return Picamera2Capture(index)
        except ImportError:
            pass
    return cv2.VideoCapture(index)


def probe_camera_indices(max_index: int = 10) -> List[int]:
    found: List[int] = []
    for i in range(max_index + 1):
        cap = video_capture_open(i)
        if not cap.isOpened():
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(i)
    return found


def validate_camera_index(index: int, width: int, height: int) -> None:
    cap = video_capture_open(index)
    try:
        if not cap.isOpened():
            _print_camera_error(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok or frame is None:
            _print_camera_error(index)
    finally:
        cap.release()


def _print_camera_error(requested: int) -> None:
    found = probe_camera_indices()
    print(
        f"Camera index {requested} failed to open or read a frame.",
        file=sys.stderr,
    )
    if found:
        print(f"Working indices on this machine: {found}", file=sys.stderr)
        print("  Run: python app.py --list-cameras", file=sys.stderr)
        print("  Or:  python app.py --choose-camera", file=sys.stderr)
    else:
        print(
            "No cameras responded. Check camera permissions and connections.",
            file=sys.stderr,
        )
    raise SystemExit(1)


def choose_camera_interactive(default_index: int = 0) -> int:
    if platform.system() == "Darwin":
        return _choose_camera_macos(default_index)

    import tkinter as tk
    from tkinter import messagebox, ttk

    indices = probe_camera_indices()
    if not indices:
        messagebox.showerror("No camera", "No working camera was found.")
        raise SystemExit(1)
    if len(indices) == 1:
        return indices[0]

    chosen: List[Optional[int]] = [None]
    win = tk.Tk()
    win.title("Choose camera")
    ttk.Label(win, text="Select which camera to use:").pack(pady=(12, 8), padx=12)
    lb = tk.Listbox(win, height=min(10, len(indices)), exportselection=False)
    for i in indices:
        lb.insert(tk.END, f"Camera {i}")
    lb.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
    pick = indices.index(default_index) if default_index in indices else 0
    lb.selection_set(pick)
    lb.activate(pick)

    def apply_selection() -> None:
        sel = lb.curselection()
        if sel:
            chosen[0] = indices[int(sel[0])]
        win.destroy()

    row = ttk.Frame(win)
    row.pack(pady=12)
    ttk.Button(row, text="OK", command=apply_selection).pack(side=tk.LEFT, padx=4)
    ttk.Button(row, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=4)
    lb.bind("<Double-1>", lambda _e: apply_selection())
    lb.bind("<Return>", lambda _e: apply_selection())
    win.mainloop()
    if chosen[0] is None:
        raise SystemExit(0)
    return chosen[0]


def _choose_camera_macos(default_index: int = 0) -> int:
    from macos_ui import alert, choose_from_list

    indices = probe_camera_indices()
    if not indices:
        alert("No working camera was found.", "Camera")
        raise SystemExit(1)
    if len(indices) == 1:
        return indices[0]
    labels = [f"Camera {i}" for i in indices]
    default_label = (
        f"Camera {default_index}" if default_index in indices else labels[0]
    )
    picked = choose_from_list(
        labels,
        "Select which camera to use",
        default_first=False,
        default_item=default_label,
    )
    if picked is None:
        raise SystemExit(0)
    return int(picked.split()[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-pose camera (train via training_app.py)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--choose-camera", action="store_true")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", type=Path, default=None)
    return p.parse_args()


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _draw_landmark_skeleton(frame: np.ndarray, lm: np.ndarray) -> None:
    h, w = frame.shape[:2]
    pts = np.column_stack((lm[:, 0] * w, lm[:, 1] * h)).astype(np.int32)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (40, 220, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 3, (0, 180, 255), -1, cv2.LINE_AA)


def run_camera_session(
    args: argparse.Namespace,
    display_imgs: Dict[str, np.ndarray],
    camera_index: int,
) -> None:
    extractor = create_extractor(model_path=args.model, prefer_mediapipe=True)
    disabled_reason = None if extractor.enabled else getattr(extractor, "reason", "Detector disabled")

    cap = video_capture_open(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    last_label: Optional[str] = None
    last_conf: float = 0.0
    frame_ms = 0

    model_ok = learned_classifier.is_loaded()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_ms += 33
            out = extractor.extract_for_video(frame, frame_ms)

            if out["feature"] is None:
                last_label = None
                last_conf = 0.0
            else:
                if model_ok:
                    last_label, last_conf = learned_classifier.predict(out["feature"])
                else:
                    last_label = None
                    last_conf = 0.0

            if out["all_landmarks"]:
                for lm in out["all_landmarks"]:
                    _draw_landmark_skeleton(frame, lm)

            panel_w = w // 2
            panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
            panel[:] = (28, 28, 30)

            if last_label and last_label != "unknown" and last_conf > 0:
                title = last_label.replace("_", " ").title()
                cv2.putText(
                    frame,
                    f"{title}  ({last_conf:.0%})",
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                if last_label in display_imgs:
                    panel[:, :] = resize_fit(display_imgs[last_label], panel_w, h)
                else:
                    cv2.putText(
                        panel,
                        title,
                        (24, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (200, 200, 200),
                        2,
                        cv2.LINE_AA,
                    )
            elif not model_ok:
                cv2.putText(
                    panel,
                    "No trained model",
                    (24, h // 2 - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 160, 120),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    "training_app.py + Train",
                    (24, h // 2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    panel,
                    "Show a hand pose",
                    (40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

            mode_txt = f"{extractor.backend_name}"
            cv2.putText(
                frame,
                mode_txt,
                (16, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 160, 200),
                1,
                cv2.LINE_AA,
            )
            if disabled_reason:
                cv2.putText(
                    frame,
                    "Detector disabled: install/fix MediaPipe",
                    (16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (20, 80, 255),
                    2,
                    cv2.LINE_AA,
                )

            combined = np.hstack([frame, panel])
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


def run() -> None:
    args = parse_args()

    if args.list_cameras:
        idx = probe_camera_indices()
        if not idx:
            print("No working cameras found.", file=sys.stderr)
            raise SystemExit(1)
        print("Working camera indices:")
        for i in idx:
            print(f"  {i}")
        return

    camera_index = args.camera
    if args.choose_camera:
        camera_index = choose_camera_interactive(default_index=args.camera)

    validate_camera_index(camera_index, args.width, args.height)

    learned_classifier.load_model(PROJECT_ROOT)
    display_imgs = load_display_images(PROJECT_ROOT)

    run_camera_session(args, display_imgs, camera_index)


if __name__ == "__main__":
    run()
