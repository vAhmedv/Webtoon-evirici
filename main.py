"""Webtoon Translator — Main Application Entry Point."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Initialize CUDA & DLL search paths before importing GUI / ML packages
from core.system.cuda_init import init_cuda_runtime
init_cuda_runtime()

from gui.app import main

if __name__ == "__main__":
    main()
