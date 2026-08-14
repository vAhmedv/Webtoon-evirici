"""Generic pre-Qwen eligibility checks for unresolved OCR regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.detection import Region


@dataclass(frozen=True)
class RepairEligibilityDecision:
    eligible: bool
    reason: str
    evidence: dict[str, object]


def evaluate_repair_eligibility(region: Region, verdict: Any) -> RepairEligibilityDecision:
    """Require independent CTD text evidence before visual OCR repair.

    A verifier-only character is not itself proof that a crop contains text.
    Rejected candidates remain REVIEW; this gate never converts them to SKIP.
    """

    metadata = region.metadata if isinstance(region.metadata, dict) else {}
    validity = metadata.get("region_validity")
    validity = validity if isinstance(validity, dict) else {}
    line_count = int(validity.get("line_polygon_count") or 0)
    segment_count = int(validity.get("segmentation_polygon_count") or 0)
    boundary_touches = int(validity.get("segmentation_boundary_touches") or 0)
    max_line_aspect = float(validity.get("max_line_aspect") or 0.0)
    evidence: dict[str, object] = {
        "validity_reason": validity.get("reason"),
        "line_polygon_count": line_count,
        "segmentation_polygon_count": segment_count,
        "segmentation_boundary_touches": boundary_touches,
        "max_line_aspect": max_line_aspect,
        "ocr_reason": getattr(verdict, "reason", None),
    }

    if validity.get("valid") is False:
        return RepairEligibilityDecision(False, "strong_region_validity_rejection", evidence)
    if not bool(getattr(verdict, "requires_review", False)) or not bool(
        getattr(verdict, "needs_repair", False)
    ):
        return RepairEligibilityDecision(False, "no_unresolved_ocr_ambiguity", evidence)

    meaningful_geometry = line_count > 0 and (segment_count > 0 or max_line_aspect >= 1.5)
    if not meaningful_geometry:
        return RepairEligibilityDecision(False, "weak_ctd_text_geometry", evidence)

    reason = str(getattr(verdict, "reason", "") or "")
    if reason == "primary_empty_verifier_filled":
        # When only the verifier produced text, require multiple independent
        # CTD lines or a supported, non-boundary-clipped elongated text shape.
        # This keeps genuine empty-primary story crops eligible without treating
        # a single PaddleOCR-VL hallucination on artwork as text evidence.
        verifier_only_geometry = (
            (line_count >= 2 and (segment_count >= 1 or max_line_aspect >= 1.5))
            or (
                line_count >= 1
                and segment_count >= 4
                and boundary_touches <= 2
                and max_line_aspect >= 1.5
            )
        )
        if not verifier_only_geometry:
            return RepairEligibilityDecision(False, "verifier_only_weak_text_geometry", evidence)

    return RepairEligibilityDecision(True, "unresolved_ocr_with_ctd_text_evidence", evidence)
