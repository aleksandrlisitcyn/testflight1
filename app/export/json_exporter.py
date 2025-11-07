import json

def export_json(pattern: dict) -> str:
    return json.dumps(pattern, ensure_ascii=False, indent=2)
