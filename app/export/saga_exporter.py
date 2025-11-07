from jinja2 import Template

TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<CrossStitchSaga>
  <CanvasGrid width="{{ grid.width }}" height="{{ grid.height }}"/>
  <ColorList>
    {% for c in palette %}
    <Color brand="{{ c.brand }}" code="{{ c.code }}" name="{{ c.name|default('') }}" symbol="{{ c.symbol|default('') }}"/>
    {% endfor %}
  </ColorList>
  <Stitches>
    {% for s in stitches %}
    <Stitch x="{{ s.x }}" y="{{ s.y }}" brand="{{ s.thread.brand }}" code="{{ s.thread.code }}"/>
    {% endfor %}
  </Stitches>
  <SymbolMap>...</SymbolMap>
</CrossStitchSaga>
"""

def export_saga(pattern: dict) -> str:
    t = Template(TEMPLATE)
    return t.render(
        grid=pattern["canvasGrid"],
        palette=pattern["palette"],
        stitches=pattern["stitches"]
    )
