from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Template


@lru_cache(maxsize=1)
def _load_template() -> Template:
    template_path = Path(__file__).resolve().parent / "templates" / "saga_template.xml"
    content = template_path.read_text(encoding="utf-8")
    return Template(content)


def export_saga(pattern: dict) -> str:
    template = _load_template()
    return template.render(
        grid=pattern["canvasGrid"],
        palette=pattern["palette"],
        stitches=pattern["stitches"],
    )
