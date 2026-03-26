"""Build k-NN pipeline from data/training and write data/model/."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from landmark_extract import create_extractor
from training_data import list_class_slugs, list_images, model_dir, training_root

PROJECT_ROOT = Path(__file__).resolve().parent


def run_training(
    project_root: Optional[Path] = None,
    hand_model_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Load all training images, extract landmarks, fit Pipeline, save classifier + metadata.
    Returns (ok, message).
    """
    root = project_root if project_root is not None else PROJECT_ROOT
    slugs = list_class_slugs(root)
    if len(slugs) < 2:
        return (
            False,
            f"Need at least 2 classes under {training_root(root)}; found {len(slugs)}.",
        )

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    skipped = 0

    extractor = create_extractor(model_path=hand_model_path, prefer_mediapipe=True)
    if not extractor.enabled:
        extractor.close()
        reason = getattr(extractor, "reason", "MediaPipe detector unavailable.")
        return (
            False,
            f"{reason} Install/fix MediaPipe and rerun training.",
        )
    try:
        for slug in slugs:
            paths = list_images(slug, root)
            if not paths:
                return (
                    False,
                    f"Class '{slug}' has no images. Add images or remove the empty folder.",
                )
            for p in paths:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    skipped += 1
                    continue
                feature = extractor.extract_training_feature(bgr)
                if feature is None:
                    skipped += 1
                    continue
                X_list.append(feature)
                y_list.append(slug)
    finally:
        extractor.close()

    if len(X_list) < 2:
        return (
            False,
            f"Not enough samples with detectable hands (got {len(X_list)}, skipped {skipped}).",
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=object)

    counts = Counter(y_list)
    if len(counts) < 2:
        return False, "Need samples from at least 2 different classes."

    n_samples = X.shape[0]
    k = min(3, max(1, n_samples))

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights="distance",
                    algorithm="auto",
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    out_dir = model_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf_path = out_dir / "classifier.joblib"
    joblib.dump(pipeline, clf_path)

    knn: KNeighborsClassifier = pipeline.named_steps["knn"]
    meta: Dict = {
        "feature_version": extractor.feature_version,
        "feature_backend": extractor.backend_name,
        "labels": knn.classes_.tolist(),
        "class_counts": {
            str(lab): int(counts.get(str(lab), 0)) for lab in knn.classes_
        },
        "n_neighbors": k,
        "skipped_images": skipped,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    msg = (
        f"Trained on {n_samples} hand samples, {len(counts)} classes, "
        f"k={k}. Skipped {skipped} images (no hand or unreadable)."
    )
    return True, msg
