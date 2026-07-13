"""Reproducible download of the Kaggle datasets into data/raw/.

Prerequisites:
    - Kaggle token configured (~/.kaggle/access_token or ~/.kaggle/kaggle.json)
    - `pip install -r requirements.txt`

Usage:
    python -m injury_risk.data.download            # download everything
    python -m injury_risk.data.download sirp-600   # a single dataset (by key)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Project root (src/injury_risk/data/download.py -> go up 2 levels)
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# local key -> Kaggle ref (owner/dataset-name)
DATASETS: dict[str, str] = {
    "epl-player-injuries": "amritbiswas007/player-injuries-and-team-performance-dataset",
    "injury-prediction-mrsimple07": "mrsimple07/injury-prediction-dataset",
    "university-football-injury": "yuanchunhong/university-football-injury-prediction-dataset",
    "sirp-600": "yuanchunhong/sirp-600-sports-injury-risk-prediction-dataset",
}


def download(key: str, ref: str) -> None:
    dest = RAW_DIR / key
    dest.mkdir(parents=True, exist_ok=True)
    print(f"==> {ref} -> {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "--unzip"],
        check=True,
    )


def main(argv: list[str]) -> int:
    keys = argv or list(DATASETS)
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        print(f"Unknown key(s): {unknown}. Available: {list(DATASETS)}")
        return 1
    for key in keys:
        download(key, DATASETS[key])
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
