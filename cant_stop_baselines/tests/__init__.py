"""Regression tests for the Can't Stop simulator and its ML components.

Run the whole suite from the project root with::

    python -m unittest discover -s tests -t . -v
"""
import sys
from pathlib import Path

# A few tests exercise helper code inside the ``scripts/`` entry points. Make
# them importable no matter which directory discovery is launched from.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
