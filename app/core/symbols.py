SYMBOLS = list("!@#$%^&*()_+=[]{};:,.<>?/\|~" + "".join(chr(0x25A0 + i) for i in range(16)))

def assign_symbols_to_palette(palette: list[dict]) -> list[dict]:
    if len(palette) > 300:
        raise ValueError("Too many unique colors (max 300)")
    result = []
    for i, c in enumerate(palette):
        c = dict(c)
        c['symbol'] = c.get('symbol') or (SYMBOLS[i % len(SYMBOLS)] if i < len(SYMBOLS) else str(i))
        result.append(c)
    return result
