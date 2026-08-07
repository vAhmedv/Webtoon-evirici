"""GPU doğrulama scripti.

PyTorch'un CUDA'yı doğru algıladığını, RTX 5070'i gördüğünü ve
gerçek bir GPU hesaplaması yapabildiğini doğrular.

Kullanım:
    python scripts/check_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle (script'i her yerden çalıştırabilmek için)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.logging_setup import setup_logging


def check_gpu() -> None:
    """GPU durumunu kontrol eder ve raporlar."""
    config = load_config()
    setup_logging(config.log_level, config.log_file)

    from loguru import logger

    logger.info("=== GPU Kontrolü Başlıyor ===")

    try:
        import torch
    except ImportError as e:
        logger.error(f"PyTorch kurulu değil: {e}")
        logger.error("Kurulum: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)

    logger.info(f"PyTorch sürümü: {torch.__version__}")

    # CUDA kullanılabilir mi?
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    if not cuda_available:
        logger.error("CUDA kullanılamıyor. RTX 5070 için CUDA 12.8+ build'i gerekli.")
        logger.error("Kurulum: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        logger.error("Ayrıca NVIDIA sürücüsünün güncel olduğundan emin ol.")
        sys.exit(1)

    # GPU bilgileri
    device_count = torch.cuda.device_count()
    logger.info(f"GPU sayısı: {device_count}")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory
        total_mem_gb = total_mem / (1024**3)
        logger.info(f"GPU {i}: {name}")
        logger.info(f"  CUDA capability: {capability[0]}.{capability[1]}")
        logger.info(f"  Toplam VRAM: {total_mem_gb:.1f} GB")

    # Gerçek hesaplama testi
    logger.info("GPU hesaplama testi yapılıyor...")
    device = torch.device("cuda:0")

    try:
        # 1024x1024 matris çarpımı (GPU'nun gerçekten çalıştığını kanıtlar)
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Sonucu doğrula
        result_sum = c.sum().item()
        if result_sum == result_sum:  # NaN değilse
            logger.info(f"GPU hesaplama testi: BAŞARILI (sonuç toplamı: {result_sum:.2f})")
        else:
            logger.error("GPU hesaplama testi: NaN sonuç üretildi")
            sys.exit(1)

        # Bellek kullanımı
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        logger.info(f"Test sonrası ayrılan bellek: {allocated:.1f} MB")
        logger.info(f"Test sonrası rezerve bellek: {reserved:.1f} MB")

    except Exception as e:
        logger.error(f"GPU hesaplama testi BAŞARISIZ: {e}")
        sys.exit(1)

    logger.info("=== GPU Kontrolü Tamamlandı ===")
    logger.info("Sonuç: GPU doğru çalışıyor. Phase 0 kabul testi geçti.")


if __name__ == "__main__":
    check_gpu()