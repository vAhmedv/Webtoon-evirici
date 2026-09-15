"""Akraba-küçük-liste deneyi: akıcı-ama-yanlış izi (sentetik, pipeline-dışı)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.translation.protection import find_kinship_mismatch
from scripts.mine_blind_spots_v1 import EMBED_SIM_THRESHOLD, judge_embed


def test_parents_to_aile_flags() -> None:
    assert find_kinship_mismatch("MY PARENTS ARE HERE", "Ailem burada") == ["kinship_ambiguous"]


def test_parents_to_ebeveyn_passes() -> None:
    assert find_kinship_mismatch("MY PARENTS ARE HERE", "Ebeveynlerim burada") == []


def test_sister_to_kizi_flags() -> None:
    assert find_kinship_mismatch("MY SISTER IS STRONG", "Kızım çok güçlü") == ["kinship_ambiguous"]


def test_sister_to_kizkardes_passes() -> None:
    assert find_kinship_mismatch("MY SISTER IS STRONG", "Kız kardeşim çok güçlü") == []
    assert find_kinship_mismatch("MY SISTER IS STRONG", "Ablam çok güçlü") == []


def test_no_kinship_no_flag() -> None:
    assert find_kinship_mismatch("THE SWORD IS SHARP", "Kılıç keskin") == []
    assert find_kinship_mismatch("", "") == []


def test_echo_no_flag() -> None:
    assert find_kinship_mismatch("SISTER", "SISTER") == []


def test_embed_threshold_calibrated() -> None:
    # Kalibrasyon (703 çift): ort=0.748 p5=0.560. Eşik p5 bandında.
    assert EMBED_SIM_THRESHOLD == 0.55
    assert judge_embed(0.40) is True
    assert judge_embed(0.75) is False
    # İnce-rol hatası YÜKSEK skorlanır (kapalı listenin işi — dürüst sınır).
    assert judge_embed(0.68) is False
