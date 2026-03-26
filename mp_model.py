"""Download MediaPipe Tasks hand model (not bundled in the mediapipe wheel)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Pinned asset; see https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "models" / "hand_landmarker.task"


def ensure_hand_landmarker_model(path: Path | None = None) -> Path:
    target = path or DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target
    print(f"Downloading hand landmarker model to {target} …")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, target)  # noqa: S310 — fixed Google URL
    if not target.is_file() or target.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Failed to download hand landmarker model. Save it manually from:\n  {HAND_LANDMARKER_URL}\n"
            f"to:\n  {target}"
        )
    print("Model ready.")
    return target
