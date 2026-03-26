"""Normalize hand landmarks to a fixed vector for training and inference."""

from __future__ import annotations

import numpy as np

FEATURE_VERSION = "norm_wrist_scale_v1"


def landmarks_to_vector(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks: (21, 3) MediaPipe order, normalized image coords + relative z.
    Translate to wrist, scale by wrist-to-middle-MCP distance, flatten to 63 dims.
    """
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected (21, 3), got {landmarks.shape}")
    wrist = landmarks[0].astype(np.float64)
    centered = landmarks.astype(np.float64) - wrist
    scale = float(np.linalg.norm(landmarks[9].astype(np.float64) - wrist))
    scale = max(scale, 1e-6)
    return (centered / scale).reshape(-1)
