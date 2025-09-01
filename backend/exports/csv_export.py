import csv
from pathlib import Path
from typing import Iterable

def to_csv(job_id: str, rows: Iterable[dict]) -> str:
    path = Path("backend/exports") / f"{job_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return str(path)
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return str(path)
