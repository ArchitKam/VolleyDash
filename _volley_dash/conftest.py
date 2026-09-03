"""
conftest.py
============
Puts this package's directory on sys.path so the tests import the
modules by their plain names regardless of the directory pytest was
started from.
"""

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)
