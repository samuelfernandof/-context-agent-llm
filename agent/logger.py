import json
from datetime import datetime

def log_event(event_type, content):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "content": content,
    }
    with open("agent.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
