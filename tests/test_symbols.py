from __future__ import annotations

from app.core.symbols import assign_symbols_to_palette


def test_assign_symbols_produces_unique_values():
    palette = [{"brand": "DMC", "code": str(i), "rgb": (i, i, i)} for i in range(60)]
    assigned = assign_symbols_to_palette(palette)
    symbols = [entry["symbol"] for entry in assigned]
    assert len(symbols) == len(set(symbols))
