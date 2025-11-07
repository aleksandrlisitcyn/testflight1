# Minimal built-in palette samples (extend with full lists)
DMC = [
    {"brand": "DMC", "code": "310", "name": "Black", "rgb": (0,0,0)},
    {"brand": "DMC", "code": "321", "name": "Red", "rgb": (199,43,59)},
    {"brand": "DMC", "code": "699", "name": "Green", "rgb": (0,92,9)},
    {"brand": "DMC", "code": "797", "name": "Blue", "rgb": (19,71,125)},
]

GAMMA = [
    {"brand": "Gamma", "code": "0001", "name": "Sample", "rgb": (10,10,10)},
]

ANCHOR = [
    {"brand": "Anchor", "code": "403", "name": "Black", "rgb": (0,0,0)},
]

def load_palette(brand: str) -> list[dict]:
    if brand == "DMC":
        return DMC
    if brand == "Gamma":
        return GAMMA
    if brand == "Anchor":
        return ANCHOR
    # auto -> default DMC
    return DMC
