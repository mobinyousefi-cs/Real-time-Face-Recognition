#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: detection.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Face detection utilities built on top of OpenCV Haar cascade classifier.

Usage:
from realtime_face_recognition.detection import load_face_detector, detect_faces

Notes:
- Uses the built-in OpenCV haarcascade files (cv2.data.haarcascades).
- Detection works on grayscale frames; callers are responsible for conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2

from .config import HAAR_CASCADE_FILENAME


Rect = Tuple[int, int, int, int]


@dataclass
class DetectionConfig:
    """Configuration for Haar cascade face detection."""

    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (30, 30)


def load_face_detector() -> cv2.CascadeClassifier:
    """Load the Haar cascade classifier for frontal face detection.

    Returns
    -------
    cv2.CascadeClassifier
        Configured Haar cascade classifier.
    """

    cascade_path = cv2.data.haarcascades + HAAR_CASCADE_FILENAME
    classifier = cv2.CascadeClassifier(cascade_path)

    if classifier.empty():  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    return classifier


def detect_faces(
    gray_frame, classifier: cv2.CascadeClassifier, config: DetectionConfig | None = None
) -> List[Rect]:
    """Detect faces in a grayscale frame.

    Parameters
    ----------
    gray_frame : numpy.ndarray
        Single-channel (grayscale) image.
    classifier : cv2.CascadeClassifier
        Loaded Haar cascade classifier.
    config : DetectionConfig | None
        Optional detection configuration. If omitted, defaults are used.

    Returns
    -------
    List[Rect]
        List of bounding boxes (x, y, w, h) for each detected face.
    """

    if config is None:
        config = DetectionConfig()

    faces = classifier.detectMultiScale(
        gray_frame,
        scaleFactor=config.scale_factor,
        minNeighbors=config.min_neighbors,
        minSize=config.min_size,
    )

    return list(faces)


def draw_boxes(
    frame,
    faces: List[Rect],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw bounding boxes for detected faces in-place on a BGR frame."""

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


__all__ = [
    "Rect",
    "DetectionConfig",
    "load_face_detector",
    "detect_faces",
    "draw_boxes",
]
