#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: cli.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Command-line interface entrypoint for the real-time face recognition project.

Usage:
python -m realtime_face_recognition.cli <command> [options]

Available commands:
- capture   : Capture face images from webcam.
- train     : Train LBPH face recognizer from captured images.
- recognize : Run real-time face recognition from webcam.

Notes:
- This module is also exposed as a console script: `realtime-face-recognition`.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional

from . import __version__
from . import capture_faces, recognize, train_model


CommandHandler = Callable[[List[str] | None], None]


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realtime-face-recognition",
        description="Real-time face detection and recognition with OpenCV (Haar + LBPH)",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"realtime-face-recognition {__version__}",
        help="Show program version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # capture
    capture_parser = subparsers.add_parser(
        "capture", help="Capture face images from webcam for a given label."
    )
    capture_faces.build_arg_parser(capture_parser)

    # train
    train_parser = subparsers.add_parser("train", help="Train LBPH model from captured faces.")
    train_model.build_arg_parser(train_parser)

    # recognize
    recognize_parser = subparsers.add_parser(
        "recognize", help="Run real-time face recognition from webcam."
    )
    recognize.build_arg_parser(recognize_parser)

    return parser


COMMAND_HANDLERS: Dict[str, CommandHandler] = {
    "capture": capture_faces.main,
    "train": train_model.main,
    "recognize": recognize.main,
}


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_root_parser()
    args, remaining = parser.parse_known_args(argv)

    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover - defensive
        parser.error(f"Unknown command: {args.command}")

    handler(remaining)


if __name__ == "__main__":  # pragma: no cover
    main()
