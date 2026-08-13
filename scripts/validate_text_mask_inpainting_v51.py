"""Run the bounded V5.1 validation set while preserving the V5 baseline."""

from __future__ import annotations

import json
from pathlib import Path

from validate_text_mask_inpainting_v5 import *  # noqa: F403
import validate_text_mask_inpainting_v5 as validation


validation.OUTPUT = validation.ROOT / "review_output" / "text_mask_validation_v5_1"


if __name__ == "__main__":
    validation.main()
    manifest_path = validation.OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    debug_path = validation.OUTPUT / "debug"
    for record in manifest:
        ordinal = int(record["ordinal"])
        block_id = int(record["block_id"])
        debug_manifest = debug_path / f"sample_{ordinal:02d}_block_{block_id}"
        # The authoritative extended metrics live in Inpainter.debug_records; main
        # writes them after this script's next revision-free bounded run.
        record["v5_baseline"] = str(
            validation.ROOT / "review_output" / "text_mask_validation_v5" /
            f"{ordinal:02d}_block_{block_id}" / "contact_sheet.png"
        )
        record["debug_dir"] = str(debug_manifest)
        sample_name = f"{ordinal:02d}_block_{block_id}"
        baseline_dir = validation.ROOT / "review_output" / "text_mask_validation_v5" / sample_name
        current_dir = validation.OUTPUT / sample_name
        pairs = []
        for label, filename in (
            ("V5 mask", "2_refined_text_mask_overlay.png"),
            ("V5 inpaint", "3_inpainted.png"),
            ("V5.1 mask", "6_mask_overlay.png"),
            ("V5.1 inpaint", "7_inpainted.png"),
        ):
            folder = baseline_dir if label.startswith("V5 ") else current_dir
            if (folder / filename).exists():
                pairs.append((label, Image.open(folder / filename).convert("RGB")))
        if pairs:
            comparison = current_dir / "v5_to_v51_comparison.png"
            validation._contact_sheet(pairs).save(comparison)
            record["v5_to_v51_comparison"] = str(comparison.relative_to(validation.OUTPUT))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
