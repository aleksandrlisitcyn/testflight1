from __future__ import annotations

from typing import Dict, Iterator, List

PRIMARY_SYMBOLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SECONDARY_SYMBOLS = list("0123456789!@#$%^&*()[]{}<>+-=;:,")
GLYPH_SYMBOLS = [chr(0x25A0 + i) for i in range(16)]


def _symbol_generator() -> Iterator[str]:
    for bucket in (PRIMARY_SYMBOLS, SECONDARY_SYMBOLS, GLYPH_SYMBOLS):
        for symbol in bucket:
            yield symbol
    idx = 1
    while True:
        yield f"#{idx}"
        idx += 1


def _normalise_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    symbol = symbol.strip()
    if not symbol:
        return None
    if len(symbol) > 3:
        return None
    return symbol


def assign_symbols_to_palette(palette: List[Dict]) -> List[Dict]:
    if len(palette) > 400:
        raise ValueError("Too many unique colors (max 400)")

    generator = _symbol_generator()
    used: set[str] = set()
    result: List[Dict] = []

    for entry in palette:
        data = dict(entry)
        symbol = _normalise_symbol(data.get("symbol"))
        if symbol and symbol not in used:
            assigned = symbol
        else:
            assigned = next(generator)
            while assigned in used:
                assigned = next(generator)
        used.add(assigned)
        data["symbol"] = assigned
        result.append(data)

    return result
