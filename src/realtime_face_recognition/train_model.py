#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: train_model.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Training script for the LBPH face recognizer using images captured to the
`data/raw_faces` directory.

Usage (module):
python -m realtime_face_recognition.cli train

Usage (direct):
python -m realtime_face_recognition.train_model

Notes:
- Expects training images to be single-channel (grayscale) PNG files, grouped
  by person under `data/raw_faces/<label>/*.png`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    FACE_IMAGE_SIZE,
    LABELS_PATH,
    LBPH_MODEL_PATH,
    RAW_FACES_DIR,
    ensure_directories,
)


def _iter_label_image_paths() -> List[Tuple[str, Path]]:
    """Return (label, image_path) pairs from the RAW_FACES_DIR tree."""

    pairs: List[Tuple[str, Path]] = []

    if not RAW_FACES_DIR.exists():
        raise FileNotFoundError(
            f"RAW_FACES_DIR does not exist: {RAW_FACES_DIR}. "
            "Did you run the capture step?"
        )

    for label_dir in RAW_FACES_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_path in sorted(label_dir.glob("*.png")):
            pairs.append((label, img_path))

    if not pairs:
        raise RuntimeError(
            f"No training images found under: {RAW_FACES_DIR}. "
            "Make sure you captured some faces first."
        )

    return pairs


def _load_training_data() -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """Load images and labels into arrays suitable for LBPH training."""

    label_to_id: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}
    next_id = 0

    images: List[np.ndarray] = []
    labels: List[int] = []

    for label, img_path in _iter_label_image_paths():
        if label not in label_to_id:
            label_to_id[label] = next_id
            id_to_label[next_id] = label
            next_id += 1

        label_id = label_to_id[label]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Failed to load image: {img_path}")
            continue

        if img.shape[::-1] != FACE_IMAGE_SIZE:
            img = cv2.resize(img, FACE_IMAGE_SIZE)

        images.append(img)
        labels.append(label_id)

    if not images:
        raise RuntimeError("No valid images loaded for training.")

    return images, labels, id_to_label


def train_lbph() -> None:
    """Train the LBPH face recognizer and persist the model + labels."""

    ensure_directories()

    images, labels, id_to_label = _load_training_data()

    print(f"[INFO] Loaded {len(images)} images across {len(id_to_label)} labels.")

    # LBPH face recognizer is available from opencv-contrib-python
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    LBPH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(LBPH_MODEL_PATH))

    with LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved LBPH model to: {LBPH_MODEL_PATH}")
    print(f"[INFO] Saved label mapping to: {LABELS_PATH}")


def build_arg_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train LBPH face recognizer.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    _ = parser.parse_args(argv)
    train_lbph()


if __name__ == "__main__":  # pragma: no cover
    main()
