"""
_parent_path.py
================
Puts the parent directory (the existing VolleyDash app) on sys.path so
this package can IMPORT the format-agnostic engine instead of forking
it: recruiting_tree (KnowledgeTree/MetricSpec/staging+diff+merge),
recruiting_operations (the Slice/Reduce/Rank/Compare cube algebra), and
recruiting_encoding (chart encoding). Those three know nothing about
CSVs -- they operate on a Player x Game x Metric cube and on a generic
MetricSpec payload+validator pair -- so a second, event-grain source
reuses them verbatim rather than duplicating them.

Needed because Streamlit puts the SCRIPT's directory on sys.path, not
its parent, so `streamlit run _volley_dash/app.py` would otherwise not
see recruiting_tree.py at all.

Importing this module is a side effect by design; import it before any
`from recruiting_* import ...` in this package.
"""

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PACKAGE_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
