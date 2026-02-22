"""Shared fixtures and helpers for buster-ripper tests."""

import sys
from pathlib import Path

# Allow importing buster_ripper directly without installation
sys.path.insert(0, str(Path(__file__).parent.parent))
