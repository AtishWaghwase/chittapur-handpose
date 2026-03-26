"""Load trained k-NN pipeline and predict class + confidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from landmark_features import FEATURE_VERSION
from landmark_extract import runtime_backend_name

PROJECT_ROOT = Path(__file__).resolve().parent

_pipeline: Any = None
_metadata: Optional[Dict] = None


def model_paths(base: Path | None = None) -> Tuple[Path, Path]:
    root = base if base is not None else PROJECT_ROOT
    d = root / "data" / "model"
    return d / "classifier.joblib", d / "metadata.json"


def load_model(project_root: Optional[Path] = None) -> bool:
    """Load classifier.joblib + metadata.json. Returns False if missing or wrong feature version."""
    global _pipeline, _metadata
    root = project_root if project_root is not None else PROJECT_ROOT
    clf_path, meta_path = model_paths(root)
    if not clf_path.is_file() or not meta_path.is_file():
        _pipeline = None
        _metadata = None
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("feature_version") != FEATURE_VERSION:
        print(
            "Model metadata feature_version does not match runtime backend. "
            "Please retrain with `python train.py`."
        )
        _pipeline = None
        _metadata = None
        return False
    if meta.get("feature_backend") and meta.get("feature_backend") != runtime_backend_name():
        print(
            "Model backend metadata differs from active runtime backend. "
            "Please retrain with `python train.py`."
        )
        _pipeline = None
        _metadata = None
        return False
    _pipeline = joblib.load(clf_path)
    _metadata = meta
    return True


def is_loaded() -> bool:
    return _pipeline is not None


def known_labels() -> List[str]:
    if _metadata is None:
        return []
    return list(_metadata.get("labels", []))


def predict(feature_vector: np.ndarray) -> Tuple[str, float]:
    """
    feature_vector: (63,) same as landmarks_to_vector output.
    Returns (label, confidence) where confidence is max class probability from k-NN.
    """
    if _pipeline is None:
        return ("unknown", 0.0)
    x = feature_vector.reshape(1, -1)
    proba = _pipeline.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    knn = _pipeline.named_steps["knn"]
    classes = knn.classes_
    label = str(classes[idx])
    return label, float(proba[idx])


def reload(project_root: Optional[Path] = None) -> bool:
    return load_model(project_root)
