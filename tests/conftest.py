"""Test konfigürasyonu."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.system.cuda_init import init_cuda_runtime
init_cuda_runtime()
