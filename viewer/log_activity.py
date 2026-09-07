"""Append one entry to the activity log the viewer displays. Usage:
  uv run python viewer/log_activity.py "<message>" [status]
status: info (default), running, done, or issue -- just affects the badge colour in the UI.
"""

import json
import sys
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "activity_log.jsonl"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: log_activity.py <message> [status]")
        raise SystemExit(1)
    message = sys.argv[1]
    status = sys.argv[2] if len(sys.argv) > 2 else "info"
    entry = {"time": time.time(), "message": message, "status": status}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
