from collections import Counter
from typing import List, Dict, Tuple

def build_legend(stitches: List[dict]) -> List[dict]:
    c = Counter((s['thread']['brand'], s['thread']['code']) for s in stitches)
    legend = []
    # Note: names/symbols will be filled from palette mapping in pipeline
    for (brand, code), count in c.items():
        legend.append({'brand': brand, 'code': code, 'count': count})
    return legend
