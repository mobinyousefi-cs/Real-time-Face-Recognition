#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Face Recognition with Python & OpenCV
File: __init__.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-08
Updated: 2025-11-08
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Package initialization for the `realtime_face_recognition` project.

This module exposes high-level metadata and convenience imports.

Usage:
from realtime_face_recognition import __version__

Notes:
- OpenCV (opencv-contrib-python) is required for LBPH face recognizer support.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "0.1.0"
