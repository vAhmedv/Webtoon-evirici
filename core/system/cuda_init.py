"""CUDA and ONNX Runtime GPU DLL Initializer.

Windows Python 3.8+ DLL yükleme kısıtlamalarını aşmak için pip ile kurulan
NVIDIA (cu12) ve PyTorch DLL dizinlerini sisteme ve PATH'e otomatik olarak kaydeder.
ONNXRuntime-GPU (CUDAExecutionProvider) motorunun 'Error 126' hatası almadan
sorunsuz ve yerel olarak çalışmasını sağlar.
"""

from __future__ import annotations

import glob
import os
import shutil
import site
import sys
from pathlib import Path
from typing import List, Set

from loguru import logger

_INITIALIZED: bool = False
_DLL_DIRECTORIES: List[str] = []


def find_cuda_dll_directories() -> List[str]:
    """Sanal ortamda ve sistemde bulunan tüm NVIDIA/CUDA ve PyTorch DLL dizinlerini tespit eder."""
    found_dirs: Set[str] = set()

    # 1. Site-packages dizinlerini topla
    search_roots: Set[str] = set()
    try:
        for sp in site.getsitepackages():
            if os.path.isdir(sp):
                search_roots.add(sp)
    except Exception:
        pass

    try:
        usp = site.getusersitepackages()
        if isinstance(usp, str) and os.path.isdir(usp):
            search_roots.add(usp)
    except Exception:
        pass

    for p in sys.path:
        if p and os.path.isdir(p) and ("site-packages" in p or "dist-packages" in p):
            search_roots.add(p)

    # 2. NVIDIA ve PyTorch DLL dizinlerini tara
    for root in search_roots:
        # nvidia/*/bin ve nvidia/*/lib
        for pattern in [
            os.path.join(root, "nvidia", "*", "bin"),
            os.path.join(root, "nvidia", "*", "lib"),
        ]:
            for d in glob.glob(pattern):
                if os.path.isdir(d):
                    found_dirs.add(os.path.abspath(d))

        # torch/lib
        torch_lib = os.path.join(root, "torch", "lib")
        if os.path.isdir(torch_lib):
            found_dirs.add(os.path.abspath(torch_lib))

    # 3. Sistem CUDA_PATH (varsa)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.isdir(cuda_path):
        for sub in ["bin", os.path.join("lib", "x64")]:
            d = os.path.join(cuda_path, sub)
            if os.path.isdir(d):
                found_dirs.add(os.path.abspath(d))

    return sorted(list(found_dirs))


def _ensure_ort_capi_dlls(dll_dirs: List[str]) -> None:
    """Eğer Windows dinamik yükleyicisi DLL'leri yine de çözümleyemezse,
    eksik olan kritik CUDA/cuDNN DLL'lerini onnxruntime/capi dizinine bağlar/kopyalar.
    """
    if sys.platform != "win32":
        return

    # onnxruntime/capi dizinini bul
    ort_capi_dir: Path | None = None
    for d in dll_dirs:
        # parent site-packages'ı kontrol et
        parent_sp = Path(d).resolve().parents[2]  # nvidia/pkg/bin -> site-packages
        cand = parent_sp / "onnxruntime" / "capi"
        if cand.is_dir():
            ort_capi_dir = cand
            break

    if ort_capi_dir is None:
        for p in sys.path:
            cand = Path(p) / "onnxruntime" / "capi"
            if cand.is_dir():
                ort_capi_dir = cand
                break

    if ort_capi_dir is None or not ort_capi_dir.is_dir():
        return

    # Kritik DLL'leri tara
    critical_prefixes = ("cublas", "cudnn", "cufft", "curand", "cudart", "nvrtc", "nvjitlink")
    for src_dir in dll_dirs:
        p_src = Path(src_dir)
        for dll_file in p_src.glob("*.dll"):
            dll_name = dll_file.name
            if any(dll_name.lower().startswith(prefix) for prefix in critical_prefixes):
                target_file = ort_capi_dir / dll_name
                if not target_file.exists():
                    try:
                        # Öncelikle hızlı ve sıfır disk maliyetli hardlink dene
                        try:
                            os.link(str(dll_file), str(target_file))
                        except Exception:
                            shutil.copy2(str(dll_file), str(target_file))
                        logger.debug("Linked CUDA DLL to ORT capi: {}", dll_name)
                    except Exception as copy_err:
                        logger.trace("Could not link DLL {}: {}", dll_name, copy_err)


def init_cuda_runtime() -> bool:
    """Windows üzerinde CUDA / ONNX Runtime GPU ortamını başlatır.

    DLL dizinlerini os.add_dll_directory() ve os.environ["PATH"]'e ekler.
    Tüm giriş noktalarında (main.py, CTD, testler) ilk sırada çağrılmalıdır.

    Returns:
        bool: En az bir CUDA DLL dizini başarıyla eklendiyse True, aksi takdirde False.
    """
    global _INITIALIZED, _DLL_DIRECTORIES
    if _INITIALIZED:
        return len(_DLL_DIRECTORIES) > 0

    _INITIALIZED = True

    if sys.platform != "win32":
        return True

    dll_dirs = find_cuda_dll_directories()
    added_count = 0

    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
            added_count += 1
        except Exception as e:
            logger.trace("Failed to add DLL directory {}: {}", d, e)

        # os.environ["PATH"]'e de ekle
        current_path = os.environ.get("PATH", "")
        if d not in current_path:
            os.environ["PATH"] = d + os.pathsep + current_path

    # ORT capi dizinini kontrol et ve eksik DLL'leri hazırla
    if dll_dirs:
        try:
            _ensure_ort_capi_dlls(dll_dirs)
        except Exception as ort_e:
            logger.trace("ORT capi DLL ensure exception: {}", ort_e)

    _DLL_DIRECTORIES = dll_dirs
    if added_count > 0:
        logger.debug("CUDA runtime initialized: {} DLL directories registered.", added_count)
        return True
    else:
        logger.trace("No CUDA DLL directories found in environment.")
        return False


def get_cuda_dll_directories() -> List[str]:
    """Kayıtlı CUDA DLL dizinleri listesini döndürür."""
    return list(_DLL_DIRECTORIES)


def is_cuda_runtime_initialized() -> bool:
    """CUDA çalışma zamanının başlatılıp başlatılmadığını döndürür."""
    return _INITIALIZED
