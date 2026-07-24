"""Tests for fetching a model from a GitHub Release.

Networkless: a mock transport serves canned release metadata and a fake asset, so the
client is exercised for real without hitting GitHub.
"""

from __future__ import annotations

import httpx
import pytest

from injury_risk import release
from injury_risk.release import ReleaseAssetNotFound, download_model, ensure_model

REPO = "owner/repo"
ASSET_BYTES = b"a pretend joblib model"


def _mock_transport(assets: list[dict], asset_body: bytes = ASSET_BYTES) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(200, json={"tag_name": "v1.2.3", "assets": assets})
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=asset_body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture
def patched_httpx(monkeypatch):
    """Route both httpx.get and httpx.stream through a mock transport."""

    def install(assets, body=ASSET_BYTES):
        transport = _mock_transport(assets, body)

        def fake_get(url, **kwargs):
            with httpx.Client(transport=transport) as c:
                return c.get(url, headers=kwargs.get("headers"))

        def fake_stream(method, url, **kwargs):
            client = httpx.Client(transport=transport)
            return client.stream(method, url)

        monkeypatch.setattr(release.httpx, "get", fake_get)
        monkeypatch.setattr(release.httpx, "stream", fake_stream)

    return install


def test_download_writes_the_asset(patched_httpx, tmp_path):
    patched_httpx(
        [{"name": "model_synthetic.joblib", "browser_download_url": "https://x/download"}]
    )
    path = download_model("synthetic", repo=REPO, dest_dir=tmp_path)
    assert path == tmp_path / "model_synthetic.joblib"
    assert path.read_bytes() == ASSET_BYTES


def test_download_picks_the_asset_for_the_track(patched_httpx, tmp_path):
    patched_httpx(
        [
            {"name": "model_real.joblib", "browser_download_url": "https://x/download"},
            {"name": "model_synthetic.joblib", "browser_download_url": "https://x/download"},
            {"name": "metrics_synthetic.json", "browser_download_url": "https://x/download"},
        ]
    )
    path = download_model("real", repo=REPO, dest_dir=tmp_path)
    assert path.name == "model_real.joblib"


def test_missing_asset_raises_with_a_helpful_message(patched_httpx, tmp_path):
    patched_httpx(
        [{"name": "metrics_synthetic.json", "browser_download_url": "https://x/download"}]
    )
    with pytest.raises(ReleaseAssetNotFound, match="model_synthetic.joblib"):
        download_model("synthetic", repo=REPO, dest_dir=tmp_path)


def test_ensure_model_is_a_noop_when_present(patched_httpx, tmp_path):
    """A baked-in or locally trained model must win, with no network call."""
    existing = tmp_path / "model_synthetic.joblib"
    existing.write_bytes(b"already here")

    # If it tried to download, the mock would overwrite the bytes; assert it did not.
    patched_httpx([], body=b"downloaded")
    path = ensure_model("synthetic", repo=REPO, dest_dir=tmp_path)
    assert path.read_bytes() == b"already here"


def test_ensure_model_downloads_when_absent(patched_httpx, tmp_path):
    patched_httpx(
        [{"name": "model_synthetic.joblib", "browser_download_url": "https://x/download"}]
    )
    path = ensure_model("synthetic", repo=REPO, dest_dir=tmp_path)
    assert path.exists()
    assert path.read_bytes() == ASSET_BYTES
