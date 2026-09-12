"""Output exporter for chapter pages with strict source overwrite protection.

Splits the global chapter canvas back into individual page images and saves
them to a separate output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from PIL import Image

from core.models import Page


def export_chapter_pages(
    pages: Sequence[Page],
    canvas: Image.Image,
    output_dir: str | Path,
) -> list[Path]:
    """Splits global canvas into individual page images and exports them safely.

    Args:
        pages: Sequence of Page objects from load_chapter.
        canvas: The full global chapter image canvas.
        output_dir: Destination directory path for output pages.

    Returns:
        List of created output page file paths.

    Raises:
        ValueError: If output path conflicts with any source image path.
    """
    output_dir = Path(output_dir).resolve()

    # Phase 3 Source Safety Guard: output dizini kaynak dosyalarla çakışmamalı.
    # Üç yasak durum: aynı yol, kaynak dosya output içinde, output kaynak
    # dizini içinde. `and/or` öncelik hatasına düşmemek için her koşul ayrı.
    source_paths = [p.path.resolve() for p in pages]
    for sp in source_paths:
        if output_dir == sp:
            raise ValueError(
                f"SOURCE OVERWRITE GUARD TRIGGERED: Output path '{output_dir}' "
                f"conflicts with source image path '{sp}'!"
            )
        try:
            if sp.is_relative_to(output_dir):
                raise ValueError(
                    f"SOURCE OVERWRITE GUARD TRIGGERED: Source '{sp}' is inside "
                    f"output '{output_dir}'. Output must be outside the source tree!"
                )
        except AttributeError:
            # Python <3.9 fallback
            try:
                sp.relative_to(output_dir)
                raise ValueError(
                    f"SOURCE OVERWRITE GUARD TRIGGERED: Source '{sp}' is inside "
                    f"output '{output_dir}'!"
                )
            except ValueError as e:
                if "TRIGGERED" in str(e):
                    raise
        try:
            if output_dir.is_relative_to(sp.parent):
                raise ValueError(
                    f"SOURCE OVERWRITE GUARD TRIGGERED: Output '{output_dir}' is inside "
                    f"source directory '{sp.parent}'. Output must be outside the source tree!"
                )
        except AttributeError:
            try:
                output_dir.relative_to(sp.parent)
                raise ValueError(
                    f"SOURCE OVERWRITE GUARD TRIGGERED: Output '{output_dir}' is inside "
                    f"source directory '{sp.parent}'!"
                )
            except ValueError as e:
                if "TRIGGERED" in str(e):
                    raise

    pages_output_dir = output_dir / "pages"
    pages_output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []
    y_cursor = 0

    for page in pages:
        y_next = y_cursor + page.height
        # Crop page region from canvas
        page_crop = canvas.crop((0, y_cursor, canvas.width, y_next))

        out_file = pages_output_dir / page.path.name
        # Double check safety for out_file path
        if out_file.resolve() in source_paths:
            raise ValueError(
                f"SOURCE OVERWRITE GUARD TRIGGERED: Cannot export to source path '{out_file}'!"
            )

        page_crop.save(out_file)
        exported_paths.append(out_file)
        y_cursor = y_next

    return exported_paths
