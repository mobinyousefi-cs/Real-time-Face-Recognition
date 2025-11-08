#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: recognize.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Real-time face recognition from webcam using a trained LBPH face recognizer.

Usage (module):
python -m realtime_face_recognition.cli recognize

Usage (direct):
python -m realtime_face_recognition.recognize

Notes:
- Requires a trained LBPH model and labels file.
- Press `q` to quit the recognition loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import cv2

from .config import LABELS_PATH, LBPH_MODEL_PATH, FACE_IMAGE_SIZE
from .detection import DetectionConfig, detect_faces, load_face_detector


def _load_labels() -> Dict[int, str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found at {LABELS_PATH}. Did you run the training step?"
        )

    with LABELS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Keys were saved as strings; convert back to int
    return {int(k): v for k, v in data.items()}


def _load_recognizer() -> cv2.face_LBPHFaceRecognizer:
    if not LBPH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LBPH model not found at {LBPH_MODEL_PATH}. Did you run the training step?"
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(LBPH_MODEL_PATH))
    return recognizer


def recognize_from_webcam(camera_index: int = 0, confidence_threshold: float = 80.0) -> None:
    """Run real-time face recognition from webcam.

    Parameters
    ----------
    camera_index : int, optional
        Webcam index (default is 0).
    confidence_threshold : float, optional
        Maximum LBPH confidence value to accept a prediction. Lower is better.
    """

    labels = _load_labels()
    recognizer = _load_recognizer()
    classifier = load_face_detector()
    detection_cfg = DetectionConfig()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():  # pragma: no cover - requires a webcam
        raise RuntimeError("Cannot open webcam. Check camera connection/index.")

    print("[INFO] Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, classifier, detection_cfg)

            for (x, y, w, h) in faces:
                face_roi = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi, FACE_IMAGE_SIZE)

                label_id, confidence = recognizer.predict(face_resized)

                if confidence <= confidence_threshold:
                    name = labels.get(label_id, "Unknown")
                    text = f"{name} ({confidence:.1f})"
                    color = (0, 255, 0)
                else:
                    text = f"Unknown ({confidence:.1f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.imshow("Real-time Face Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:  # pragma: no cover - requires display/webcam
        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Real-time face recognition from webcam.")

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index (default: 0).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=80.0,
        help="Maximum LBPH confidence to accept a prediction (lower is better).",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    recognize_from_webcam(
        camera_index=args.camera_index,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
