#!/usr/bin/env python3
"""CLI: train k-NN hand-pose classifier from data/training/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training_fit import run_training

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Train hand-pose classifier from data/training/")
    p.add_argument(
        "command",
        nargs="?",
        default="fit",
        choices=("fit",),
        help="fit — train from images (default)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to hand_landmarker.task",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (default: repo root)",
    )
    args = p.parse_args()

    ok, msg = run_training(project_root=args.root, hand_model_path=args.model)
    print(msg)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
