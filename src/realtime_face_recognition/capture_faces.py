#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: capture_faces.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Script utilities for capturing face images from a webcam and storing them on
disk as a training dataset.

Usage (module):
python -m realtime_face_recognition.cli capture --label mobin

Usage (direct):
python -m realtime_face_recognition.capture_faces --label mobin

Notes:
- Press `c` to capture the current detected face.
- Press `q` to quit the capture loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2

from .config import FACE_IMAGE_SIZE, RAW_FACES_DIR, ensure_directories
from .detection import DetectionConfig, detect_faces, load_face_detector


def _create_person_dir(label: str) -> Path:
    person_dir = RAW_FACES_DIR / label
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


def capture_from_webcam(label: str, camera_index: int = 0) -> None:
    """Capture face images from webcam for a given label.

    Parameters
    ----------
    label : str
        Name/identifier of the person (used as folder name).
    camera_index : int, optional
        Webcam index (default is 0).
    """

    ensure_directories()
    person_dir = _create_person_dir(label)

    classifier = load_face_detector()
    detection_cfg = DetectionConfig()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():  # pragma: no cover - requires a webcam
        raise RuntimeError("Cannot open webcam. Check camera connection/index.")

    print("[INFO] Press 'c' to capture a face, 'q' to quit.")

    counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, classifier, detection_cfg)

            # Draw boxes and show instructions
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                "Press 'c' to capture, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Face Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and faces:
                # For simplicity, use the first detected face
                (x, y, w, h) = faces[0]
                face_roi = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi, FACE_IMAGE_SIZE)

                filename = person_dir / f"{label}_{counter:04d}.png"
                cv2.imwrite(str(filename), face_resized)
                counter += 1
                print(f"[INFO] Saved: {filename}")

    finally:  # pragma: no cover - requires a display/webcam
        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Create an argument parser for the capture script."""

    if parser is None:
        parser = argparse.ArgumentParser(description="Capture face images from webcam.")

    parser.add_argument(
        "--label",
        required=True,
        help="Person label/name used as folder name under data/raw_faces.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index (default: 0).",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    capture_from_webcam(label=args.label, camera_index=args.camera_index)


if __name__ == "__main__":  # pragma: no cover
    main()
