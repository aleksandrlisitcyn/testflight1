from app.color.palette_loader import load_palette
from app.color.palette_matcher import build_kd, nearest_color

def test_nearest_color():
    pal = load_palette("DMC")
    kd = build_kd(pal)
    c = nearest_color(kd, [0,0,0], pal)
    assert c["code"] in {"310"}  # black-like
