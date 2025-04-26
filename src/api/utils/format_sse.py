import json


def format_sse(data: dict, field: str = "data") -> str:
    """Formats a dictionary as a Server-Sent Event."""
    json_data = json.dumps(data)
    return f"{field}: {json_data}\n\n"
