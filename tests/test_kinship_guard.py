"""Akraba-küçük-liste deneyi: akıcı-ama-yanlış izi (sentetik, pipeline-dışı)."""

from core.translation.protection import find_kinship_mismatch


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
