import io
import json
import zipfile
import xml.etree.ElementTree as ET

from app.export.css_exporter import export_css
from app.export.dize_exporter import export_dize
from app.export.xsd_exporter import export_xsd
from app.export.xsp_exporter import export_xsp


DUMMY_PATTERN = {
    "canvasGrid": {"width": 2, "height": 2},
    "palette": [
        {"brand": "DMC", "code": "321", "name": "Red", "rgb": [200, 32, 48], "symbol": "★"},
        {"brand": "DMC", "code": "310", "name": "Black", "rgb": [0, 0, 0], "symbol": "■"},
    ],
    "stitches": [
        {"x": 0, "y": 0, "thread": {"brand": "DMC", "code": "321"}},
        {"x": 1, "y": 0, "thread": {"brand": "DMC", "code": "310"}},
        {"x": 0, "y": 1, "thread": {"brand": "DMC", "code": "321"}},
    ],
    "meta": {"title": "Sample", "brand": "DMC"},
}


def test_export_xsd_structure():
    xml_payload = export_xsd(DUMMY_PATTERN)
    root = ET.fromstring(xml_payload)
    assert root.tag == "CrossStitchDesign"
    canvas = root.find("Canvas")
    assert canvas is not None
    assert canvas.get("width") == "2"
    color = root.find("Legend/Color")
    assert color is not None
    assert color.get("code") == "321"


def test_export_xsp_contains_hex():
    payload = json.loads(export_xsp(DUMMY_PATTERN))
    assert payload["canvas"]["width"] == 2
    assert payload["legend"][0]["hex"].startswith("#")


def test_export_css_has_sections():
    css = export_css(DUMMY_PATTERN)
    assert "[Legend]" in css
    assert "★" in css
    assert "[Stitches]" in css


def test_export_dize_zip_contents():
    data = export_dize(DUMMY_PATTERN)
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        names = set(archive.namelist())
        assert {"pattern.json", "legend.csv", "stitches.csv"}.issubset(names)
        legend = archive.read("legend.csv").decode("utf-8")
        assert "321" in legend
