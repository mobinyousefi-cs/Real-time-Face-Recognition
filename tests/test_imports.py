#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: test_imports.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Minimal smoke tests to verify that the package imports correctly.

Usage:
pytest tests/test_imports.py
"""

from __future__ import annotations

import importlib


def test_import_package() -> None:
    importlib.import_module("realtime_face_recognition")


def test_import_submodules() -> None:
    for name in [
        "realtime_face_recognition.config",
        "realtime_face_recognition.detection",
        "realtime_face_recognition.capture_faces",
        "realtime_face_recognition.train_model",
        "realtime_face_recognition.recognize",
        "realtime_face_recognition.cli",
    ]:
        importlib.import_module(name)
