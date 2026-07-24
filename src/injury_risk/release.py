"""Fetch a trained model from a GitHub Release, instead of training locally.

Trained models are gitignored — they are build artefacts, not source. But a deployed
app needs one, and training on a small free host is slow and non-deterministic across
environments. So the release pipeline attaches the delivered model to each GitHub
Release (see .github/workflows/release.yml), and this fetches it: a lightweight model
registry, no MLflow needed at this scale.

The deployment (and `injury-risk fetch-model`) call :func:`ensure_model`, which is a
no-op when the model is already present — so a container that baked one in, or a repo
checkout that trained one, does not re-download.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from injury_risk.config import MODELS_DIR

DEFAULT_REPO = "Amayyas/athlete-injury-risk-detection"
GITHUB_API = "https://api.github.com"


class ReleaseAssetNotFound(RuntimeError):
    """No matching model asset on the targeted release."""


def _model_asset(track: str) -> str:
    return f"model_{track}.joblib"


def _latest_release_assets(repo: str, token: str | None) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = httpx.get(f"{GITHUB_API}/repos/{repo}/releases/latest", headers=headers, timeout=15)
    resp.raise_for_status()
    assets: list[dict] = resp.json().get("assets", [])
    return assets


def download_model(
    track: str = "synthetic",
    repo: str = DEFAULT_REPO,
    dest_dir: Path = MODELS_DIR,
    token: str | None = None,
) -> Path:
    """Download the delivered model for a track from the repo's latest release."""
    wanted = _model_asset(track)
    assets = _latest_release_assets(repo, token)
    match = next((a for a in assets if a.get("name") == wanted), None)
    if match is None:
        available = ", ".join(a.get("name", "?") for a in assets) or "none"
        raise ReleaseAssetNotFound(
            f"no asset {wanted!r} on the latest release of {repo} (assets: {available})"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / wanted
    with httpx.stream("GET", match["browser_download_url"], follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    return dest


def ensure_model(
    track: str = "synthetic",
    repo: str = DEFAULT_REPO,
    dest_dir: Path = MODELS_DIR,
    token: str | None = None,
) -> Path:
    """Return the local model path, downloading it from the latest release if absent.

    A no-op when the model already exists — a baked-in or locally trained model wins,
    and nothing hits the network.
    """
    local = dest_dir / _model_asset(track)
    if local.exists():
        return local
    return download_model(track, repo=repo, dest_dir=dest_dir, token=token)
