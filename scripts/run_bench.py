from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.pipeline import process_image_to_pattern, render_preview  # noqa: E402


def run_case(image_path: Path, output_dir: Path, pattern_mode: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    start = time.perf_counter()
    pattern = process_image_to_pattern(
        arr,
        brand="DMC",
        min_cells=25,
        detail_level="medium",
        pattern_mode=pattern_mode,  # neutral detail level, mode controls behaviour
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    color_preview = render_preview(pattern, mode="color")
    symbol_preview = render_preview(pattern, mode="symbols")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{image_path.stem}_color.png").write_bytes(color_preview)
    (output_dir / f"{image_path.stem}_symbols.png").write_bytes(symbol_preview)

    return {
        "case": image_path.name,
        "grid": {
            "width": pattern.canvasGrid.width,
            "height": pattern.canvasGrid.height,
        },
        "palette_size": len(pattern.palette),
        "total_stitches": pattern.meta.get("total_stitches"),
        "time_ms": round(elapsed_ms, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality bench on stitched cases")
    parser.add_argument("--cases", type=Path, default=Path("data/bench/cases"))
    parser.add_argument("--output", type=Path, default=Path("data/bench/bench.json"))
    parser.add_argument("--results", type=Path, default=Path("data/bench/results"))
    parser.add_argument(
        "--pattern-mode",
        choices=["color", "symbol", "mixed"],
        default="color",
        help="Chart type (affects sampling heuristics)",
    )
    args = parser.parse_args()

    cases = sorted(p for p in args.cases.glob("*.png"))
    results = []
    for path in cases:
        stats = run_case(path, args.results, args.pattern_mode)
        results.append(stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
