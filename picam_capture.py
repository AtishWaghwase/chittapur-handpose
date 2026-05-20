"""Picamera2 wrapper with a cv2.VideoCapture-compatible interface for RPi CSI cameras."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# Suppress verbose libcamera INFO logs
os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")


def _num_picamera2_cameras() -> int:
    try:
        from picamera2 import Picamera2
        return len(Picamera2.global_camera_info())
    except Exception:
        return 0


class Picamera2Capture:
    """Drop-in replacement for cv2.VideoCapture using picamera2 for CSI cameras."""

    def __init__(self, camera_index: int = 0) -> None:
        self._index = camera_index
        self._pc2 = None
        self._started = False
        self._w = 640
        self._h = 480
        self._valid = camera_index < _num_picamera2_cameras()

    def isOpened(self) -> bool:
        return self._valid

    def set(self, prop: int, val: float) -> bool:
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._w = int(val)
            return True
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self._h = int(val)
            return True
        return False

    def get(self, prop: int) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0

    def _ensure_started(self) -> bool:
        if self._started:
            return True
        if not self._valid:
            return False
        try:
            from picamera2 import Picamera2
            self._pc2 = Picamera2(self._index)
            config = self._pc2.create_video_configuration(
                main={"size": (self._w, self._h), "format": "BGR888"}
            )
            self._pc2.configure(config)
            self._pc2.start()
            self._started = True
            return True
        except Exception as e:
            print(f"[picam] failed to start camera {self._index}: {e}")
            self._valid = False
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._ensure_started():
            return False, None
        try:
            frame = self._pc2.capture_array("main")
            return True, frame
        except Exception:
            return False, None

    def release(self) -> None:
        if self._started and self._pc2 is not None:
            try:
                self._pc2.stop()
                self._pc2.close()
            except Exception:
                pass
            self._started = False
            self._pc2 = None
