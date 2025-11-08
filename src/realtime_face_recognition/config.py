#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Central configuration for paths and runtime parameters used across the
`realtime_face_recognition` package.

Usage:
from realtime_face_recognition.config import DATA_DIR, MODELS_DIR

Notes:
- Paths are defined relative to the project root so the code works both when
  installed as a package and when run from a local clone.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

# This file lives in: <project_root>/src/realtime_face_recognition/config.py
# parents[0] = .../src/realtime_face_recognition
# parents[1] = .../src
# parents[2] = .../project_root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_FACES_DIR: Path = DATA_DIR / "raw_faces"
PROCESSED_FACES_DIR: Path = DATA_DIR / "processed_faces"

MODELS_DIR: Path = PROJECT_ROOT / "models"
LBPH_MODEL_PATH: Path = MODELS_DIR / "lbph_face_recognizer.xml"
LABELS_PATH: Path = MODELS_DIR / "labels.json"

# Haar cascade configuration
HAAR_CASCADE_FILENAME: str = "haarcascade_frontalface_default.xml"

# Face pre-processing configuration
FACE_IMAGE_SIZE = (200, 200)  # (width, height) for training images

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------


def ensure_directories() -> None:
    """Create all required directories if they do not exist.

    This function is intentionally idempotent and safe to call many times.
    """

    for path in (DATA_DIR, RAW_FACES_DIR, PROCESSED_FACES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_FACES_DIR",
    "PROCESSED_FACES_DIR",
    "MODELS_DIR",
    "LBPH_MODEL_PATH",
    "LABELS_PATH",
    "HAAR_CASCADE_FILENAME",
    "FACE_IMAGE_SIZE",
    "ensure_directories",
]
