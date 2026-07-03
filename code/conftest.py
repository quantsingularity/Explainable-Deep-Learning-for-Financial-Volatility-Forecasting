"""
pytest conftest.py: ensures the code/ root is on sys.path so all packages
(core, training, evaluation, …) are importable when running:
    cd code && python -m pytest tests/
"""

import os
import sys

# Insert the directory containing this file (i.e. code/) at the front of sys.path
sys.path.insert(0, os.path.dirname(__file__))
