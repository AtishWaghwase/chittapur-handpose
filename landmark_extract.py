"""MediaPipe solutions.hands extractor with disabled fallback mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from landmark_features import landmarks_to_vector

MP_FEATURE_VERSION = "norm_wrist_scale_v1"
BACKEND_NAME = "mediapipe_solutions"


def runtime_feature_version() -> str:
    return MP_FEATURE_VERSION


def runtime_backend_name() -> str:
    return BACKEND_NAME


class UnifiedExtractor:
    backend_name: str
    feature_version: str
    enabled: bool

    def close(self) -> None: ...
    def extract_training_feature(self, bgr: np.ndarray) -> Optional[np.ndarray]: ...
    def extract_for_video(self, bgr: np.ndarray, frame_ms: int) -> Dict[str, Any]: ...


class DisabledExtractor:
    def __init__(self, reason: str) -> None:
        self.backend_name = BACKEND_NAME
        self.feature_version = MP_FEATURE_VERSION
        self.enabled = False
        self.reason = reason

    def close(self) -> None:
        return None

    def extract_training_feature(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        _ = bgr
        return None

    def extract_for_video(self, bgr: np.ndarray, frame_ms: int) -> Dict[str, Any]:
        _ = bgr
        _ = frame_ms
        return {
            "feature": None,
            "selected_landmarks": None,
            "all_landmarks": [],
            "quality": {},
            "disabled_reason": self.reason,
        }


class MediaPipeSolutionsExtractor:
    def __init__(self) -> None:
        import mediapipe as mp

        self.backend_name = BACKEND_NAME
        self.feature_version = MP_FEATURE_VERSION
        self.enabled = True
        self._mp = mp
        self._hands_static = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands_video = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._hands_static.close()
        self._hands_video.close()

    def extract_training_feature(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        lm = _extract_largest_landmarks_with_solutions(bgr, self._hands_static)
        if lm is None:
            return None
        return landmarks_to_vector(lm)

    def extract_for_video(self, bgr: np.ndarray, frame_ms: int) -> Dict[str, Any]:
        _ = frame_ms
        lm_all = _extract_all_landmarks_with_solutions(bgr, self._hands_video)
        selected = _select_largest_landmarks(lm_all, bgr.shape[1], bgr.shape[0]) if lm_all else None
        return {
            "feature": landmarks_to_vector(selected) if selected is not None else None,
            "selected_landmarks": selected,
            "all_landmarks": lm_all,
            "quality": {},
        }


def create_extractor(
    model_path: Path | None = None,
    prefer_mediapipe: bool = True,
) -> UnifiedExtractor:
    _ = model_path
    _ = prefer_mediapipe
    try:
        ext = MediaPipeSolutionsExtractor()
        print("[backend] using mediapipe_solutions")
        return ext
    except Exception as exc:  # pragma: no cover
        reason = (
            "MediaPipe solutions.hands could not initialize. "
            "Install/fix MediaPipe on Raspberry Pi (e.g., mediapipe-rpi4/mediapipe-rpi3). "
            f"Details: {exc}"
        )
        print(f"[backend] detector disabled: {reason}")
        return DisabledExtractor(reason=reason)


def create_image_hand_landmarker(model_path: Path | None = None):
    _ = model_path
    raise RuntimeError("Not used in solutions-only mode")


def extract_largest_hand_landmarks(
    bgr: np.ndarray,
    landmarker,
) -> Optional[np.ndarray]:
    _ = landmarker
    return _extract_largest_landmarks_with_solutions(bgr, None)


def _extract_all_landmarks_with_solutions(bgr: np.ndarray, hands_obj) -> list[np.ndarray]:
    if bgr is None or bgr.size == 0:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = hands_obj.process(rgb)
    if not res.multi_hand_landmarks:
        return []
    out: list[np.ndarray] = []
    for hl in res.multi_hand_landmarks:
        arr = np.array([[p.x, p.y, p.z] for p in hl.landmark], dtype=np.float64)
        if arr.shape == (21, 3):
            out.append(arr)
    return out


def _select_largest_landmarks(
    landmarks_list: list[np.ndarray],
    image_width: int,
    image_height: int,
) -> Optional[np.ndarray]:
    if not landmarks_list:
        return None
    def _area(lm: np.ndarray) -> float:
        xs = lm[:, 0] * image_width
        ys = lm[:, 1] * image_height
        return float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    return max(landmarks_list, key=_area)


def _extract_largest_landmarks_with_solutions(bgr: np.ndarray, hands_obj) -> Optional[np.ndarray]:
    if hands_obj is None:
        return None
    all_lm = _extract_all_landmarks_with_solutions(bgr, hands_obj)
    if not all_lm:
        return None
    h, w = bgr.shape[:2]
    return _select_largest_landmarks(all_lm, w, h)
