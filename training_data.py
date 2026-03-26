"""Slug-based training and display image layout under data/."""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def training_root(base: Path | None = None) -> Path:
    root = base if base is not None else PROJECT_ROOT
    return root / "data" / "training"


def display_root(base: Path | None = None) -> Path:
    root = base if base is not None else PROJECT_ROOT
    return root / "data" / "display"


def model_dir(base: Path | None = None) -> Path:
    root = base if base is not None else PROJECT_ROOT
    return root / "data" / "model"


def sanitize_slug(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("Class name must contain letters or numbers")
    if s.startswith("."):
        raise ValueError("Invalid class name")
    return s[:80]


def list_class_slugs(base: Path | None = None) -> List[str]:
    root = training_root(base)
    if not root.is_dir():
        return []
    slugs: List[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith((".", "_")):
            slugs.append(p.name)
    return slugs


def class_dir(slug: str, base: Path | None = None) -> Path:
    d = training_root(base) / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_class(slug: str, base: Path | None = None) -> Path:
    """Create empty class folder; slug must be pre-sanitized."""
    if slug.startswith((".", "_")):
        raise ValueError("Invalid slug")
    return class_dir(slug, base)


def _unique_name(src: Path, dest_dir: Path) -> Path:
    stem = src.stem
    suf = src.suffix.lower()
    if suf not in ALLOWED_SUFFIXES:
        suf = ".jpg"
    h = hashlib.sha256(f"{time.time_ns()}:{src}".encode()).hexdigest()[:8]
    name = f"{stem}_{int(time.time())}_{h}{suf}"
    return dest_dir / name


def import_images(
    slug: str, paths: List[str | Path], base: Path | None = None
) -> List[Path]:
    out: List[Path] = []
    dest = class_dir(slug, base)
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        target = _unique_name(path, dest)
        shutil.copy2(path, target)
        out.append(target)
    return out


def list_images(slug: str, base: Path | None = None) -> List[Path]:
    d = training_root(base) / slug
    if not d.is_dir():
        return []
    files: List[Path] = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES:
            files.append(p)
    return files


def delete_image(path: Path, base: Path | None = None) -> None:
    b = training_root(base).resolve()
    path = path.resolve()
    path.relative_to(b)
    if path.is_file():
        path.unlink()


def delete_class(slug: str, base: Path | None = None) -> None:
    d = training_root(base) / slug
    if d.is_dir():
        shutil.rmtree(d)
    disp = display_root(base) / f"{slug}.png"
    if disp.is_file():
        disp.unlink()


def set_display_image(slug: str, src: Path, base: Path | None = None) -> Path:
    display_root(base).mkdir(parents=True, exist_ok=True)
    dest = display_root(base) / f"{slug}.png"
    shutil.copy2(src, dest)
    return dest


def copy_first_training_as_display(slug: str, base: Path | None = None) -> Path | None:
    imgs = list_images(slug, base)
    if not imgs:
        return None
    return set_display_image(slug, imgs[0], base)
