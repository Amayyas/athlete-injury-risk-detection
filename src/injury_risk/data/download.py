"""Reproducible download of the Kaggle datasets into data/raw/.

Prerequisites:
    - Kaggle token configured (~/.kaggle/access_token or ~/.kaggle/kaggle.json)
    - `pip install -r requirements.txt`

"""

from __future__ import annotations

import subprocess
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
