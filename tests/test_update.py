# tests/test_update.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the complete update mechanism (service, transport, installer,
# CLI, and GUI), kept in one place so shared helpers stay together.

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.error
from collections.abc import Mapping
from pathlib import Path
from typing import Self
from unittest.mock import patch

import pytest
from packaging.version import Version
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from topoptcomec.cli.cli import run_cli
from topoptcomec.cli.update import (
    EXIT_CONFIRMATION_REQUIRED,
    EXIT_ERROR,
    EXIT_MANUAL_ACTION,
    EXIT_OK,
    run_update_command,
)
from topoptcomec.gui.main_window import MainWindow
from topoptcomec.update import (
    ChecksumError,
    DownloadError,
    GithubReleaseClient,
    Installation,
    InstallationError,
    InstallationKind,
    NetworkError,
    RateLimitError,
    ReleaseAsset,
    ReleaseInfo,
    ReleaseMetadataError,
    UpdateCheck,
    UpdatePlan,
    UpdateResult,
    UpdateStatus,
    VersionError,
    build_plan,
    check_for_update,
    detect_installation,
    determine_status,
    download_asset,
    install_update,
    parse_version_tag,
)
from topoptcomec.update.transport import UrllibTransport
from topoptcomec.version import __version__

# --- Version comparison and GitHub release checks ---


class FakeTransport:
    """In-memory transport that never touches the network."""

    def __init__(
        self,
        payload: object | None = None,
        *,
        status: int = 200,
        body: bytes | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status = status
        self.body = body
        self.error = error
        self.requests: list[tuple[str, float, dict[str, str]]] = []

    def get_json(
        self, url: str, timeout: float, headers: Mapping[str, str] | None = None
    ) -> tuple[int, Mapping[str, str], bytes]:
        self.requests.append((url, timeout, dict(headers or {})))
        if self.error is not None:
            raise self.error
        body = self.body
        if body is None:
            body = json.dumps(self.payload).encode("utf-8")
        return self.status, {}, body

    def open_stream(self, url: str, timeout: float) -> object:
        raise AssertionError("downloads are not expected in these tests")


def release_payload(**overrides: object) -> dict:
    """Return a valid GitHub release payload."""
    payload: dict = {
        "tag_name": "v3.1",
        "name": "TopoptComec v3.1",
        "body": "## What's new\n\n- Faster optimizer",
        "html_url": ("https://github.com/ninja7v/TopoptComec/releases/tag/v3.1"),
        "published_at": "2026-10-01T10:00:00Z",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": "TopoptComec-3.1-linux-x86_64.zip",
                "browser_download_url": (
                    "https://github.com/ninja7v/TopoptComec/releases/download/"
                    "v3.1/TopoptComec-3.1-linux-x86_64.zip"
                ),
                "size": 1024,
                "digest": "sha256:" + "a" * 64,
            }
        ],
    }
    payload.update(overrides)
    return payload


def checked(payload: dict, installed: str = "3.0"):
    return check_for_update(
        installed, client=GithubReleaseClient(FakeTransport(payload))
    )


@pytest.mark.parametrize(
    ("installed", "latest", "expected"),
    [
        ("3.0", "3.0", UpdateStatus.UP_TO_DATE),
        ("3.0", "3.1", UpdateStatus.UPDATE_AVAILABLE),
        ("3.1", "3.0", UpdateStatus.INSTALLED_AHEAD),
        ("2.9", "3.0", UpdateStatus.UPDATE_AVAILABLE),
        ("3.0", "3.1rc1", UpdateStatus.UP_TO_DATE),
        ("3.0", "4.0a1", UpdateStatus.UP_TO_DATE),
    ],
)
def test_determine_status(installed: str, latest: str, expected: UpdateStatus):
    """Version comparison is semantic and ignores prereleases."""
    assert determine_status(Version(installed), Version(latest)) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3.0", "3.0"),
        ("v3.0", "3.0"),
        ("V3.1.2", "3.1.2"),
        (" v2.0 ", "2.0"),
        ("3.1-rc1", "3.1rc1"),
    ],
)
def test_parse_version_tag(raw: str, expected: str):
    """Tags may carry a leading v and surrounding whitespace."""
    assert parse_version_tag(raw) == Version(expected)


@pytest.mark.parametrize("raw", ["", "v", "not-a-version", "1..2"])
def test_parse_version_tag_rejects_invalid_values(raw: str):
    """Invalid version strings raise a typed error."""
    with pytest.raises(VersionError):
        parse_version_tag(raw)


def test_parse_version_tag_rejects_non_string():
    """Non-string versions are rejected without crashing."""
    with pytest.raises(VersionError):
        parse_version_tag(3)  # type: ignore[arg-type]


def test_check_reports_available_update_with_release_details():
    """A newer stable release produces structured update information."""
    check = checked(release_payload())

    assert check.update_available
    assert check.status is UpdateStatus.UPDATE_AVAILABLE
    assert check.installed_version == Version("3.0")
    assert check.latest_version == Version("3.1")
    assert check.release.tag == "v3.1"
    assert check.release.name == "TopoptComec v3.1"
    assert "Faster optimizer" in check.release.notes
    assert check.release.page_url.endswith("/releases/tag/v3.1")
    assert len(check.release.assets) == 1
    asset = check.release.assets[0]
    assert asset.size == 1024
    assert asset.sha256 == "a" * 64


def test_check_reports_up_to_date_for_same_version():
    """Equal versions do not trigger an update."""
    check = checked(release_payload(tag_name="v3.0"), installed="3.0")
    assert not check.update_available
    assert check.status is UpdateStatus.UP_TO_DATE
    assert check.latest_version == Version("3.0")


def test_check_reports_installed_version_ahead():
    """A development version newer than the release is not downgraded."""
    check = checked(release_payload(tag_name="v3.0"), installed="3.1")
    assert not check.update_available
    assert check.status is UpdateStatus.INSTALLED_AHEAD


def test_check_rejects_invalid_installed_version_without_network():
    """An invalid installed version fails before any request is made."""
    transport = FakeTransport(release_payload())
    with pytest.raises(VersionError):
        check_for_update("nope", client=GithubReleaseClient(transport))
    assert transport.requests == []


def test_client_sends_no_auth_token_and_a_useful_user_agent():
    """Public release checks must not depend on a GitHub token."""
    transport = FakeTransport(release_payload())
    GithubReleaseClient(transport, timeout=7).latest_release()

    url, timeout, headers = transport.requests[0]
    assert url.endswith("/releases/latest")
    assert timeout == 7
    assert "Authorization" not in headers
    assert headers["User-Agent"].startswith("TopoptComec/")


def test_release_page_url_falls_back_when_metadata_is_untrusted():
    """A non-GitHub page URL is replaced by the canonical release URL."""
    check = checked(release_payload(html_url="https://evil.example/release"))
    assert check.release.page_url == (
        "https://github.com/ninja7v/TopoptComec/releases/tag/v3.1"
    )


def test_prerelease_metadata_is_ignored():
    """The updater never installs a prerelease."""
    with pytest.raises(ReleaseMetadataError):
        checked(release_payload(prerelease=True))
    with pytest.raises(ReleaseMetadataError):
        checked(release_payload(tag_name="v3.2-rc1"))


def test_draft_release_is_rejected():
    """Draft releases are not installable."""
    with pytest.raises(ReleaseMetadataError):
        checked(release_payload(draft=True))


def test_network_errors_are_propagated():
    """Network failures surface as typed errors."""
    transport = FakeTransport(error=NetworkError("timed out"))
    with pytest.raises(NetworkError, match="timed out"):
        GithubReleaseClient(transport).latest_release()


def test_rate_limit_error_is_preserved():
    """GitHub rate limiting has its own error type."""
    transport = FakeTransport(error=RateLimitError("rate limited"))
    with pytest.raises(RateLimitError):
        GithubReleaseClient(transport).latest_release()


def test_unexpected_http_status_is_rejected():
    """A non-200 response is malformed metadata, not an update."""
    with pytest.raises(ReleaseMetadataError):
        GithubReleaseClient(FakeTransport({}, status=500)).latest_release()


def test_invalid_json_body_is_rejected():
    """A body that is not JSON is rejected."""
    transport = FakeTransport(body=b"<html>not json</html>")
    with pytest.raises(ReleaseMetadataError):
        GithubReleaseClient(transport).latest_release()


def test_non_object_payload_is_rejected():
    """A JSON array is not a release."""
    transport = FakeTransport(body=b"[]")
    with pytest.raises(ReleaseMetadataError):
        GithubReleaseClient(transport).latest_release()


@pytest.mark.parametrize("tag", ["", "   ", "bad tag", "v3.0;rm -rf /"])
def test_invalid_tags_are_rejected(tag: str):
    """Tags that are empty or unsafe for git are rejected."""
    with pytest.raises((ReleaseMetadataError, VersionError)):
        checked(release_payload(tag_name=tag))


def test_invalid_version_in_tag_is_rejected():
    """A syntactically safe but non-version tag is rejected."""
    with pytest.raises(VersionError):
        checked(release_payload(tag_name="release-latest"))


@pytest.mark.parametrize(
    "payload",
    [
        release_payload(tag_name=None),
        release_payload(assets="none"),
        release_payload(assets=[None]),
        release_payload(assets=[{"browser_download_url": "https://github.com/x"}]),
        release_payload(
            assets=[
                {
                    "name": "../escape.zip",
                    "browser_download_url": "https://github.com/x.zip",
                    "size": 1,
                }
            ]
        ),
        release_payload(
            assets=[
                {
                    "name": "x.zip",
                    "browser_download_url": "http://github.com/x.zip",
                    "size": 1,
                }
            ]
        ),
        release_payload(
            assets=[
                {
                    "name": "x.zip",
                    "browser_download_url": "https://evil.example/x.zip",
                    "size": 1,
                }
            ]
        ),
        release_payload(
            assets=[
                {
                    "name": "x.zip",
                    "browser_download_url": "https://github.com/x.zip",
                    "size": -1,
                }
            ]
        ),
        release_payload(
            assets=[
                {
                    "name": "x.zip",
                    "browser_download_url": "https://github.com/x.zip",
                    "size": 1,
                    "digest": "md5:abcd",
                }
            ]
        ),
    ],
)
def test_malformed_or_untrusted_asset_metadata_is_rejected(payload: dict):
    """Incomplete or untrusted asset metadata is never accepted."""
    with pytest.raises(ReleaseMetadataError):
        checked(payload)


def test_release_without_assets_is_valid():
    """Releases are allowed to publish no artifacts; plans explain the rest."""
    check = checked(release_payload(assets=[]))
    assert check.update_available
    assert check.release.assets == ()


def test_client_rejects_non_positive_timeout():
    """A nonsensical timeout is caught at construction time."""
    with pytest.raises(ValueError):
        GithubReleaseClient(timeout=0)


# --- HTTP transport error mapping ---


class FakeResponse:
    """Minimal urlopen response."""

    def __init__(self, body: bytes = b"{}", status: int = 200) -> None:
        self._body = body
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self.url = "https://api.github.com/repos/ninja7v/TopoptComec/releases/latest"
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True

    def geturl(self) -> str:
        return self.url

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()


def http_error(code: int, *, remaining: str | None = None) -> urllib.error.HTTPError:
    headers = {} if remaining is None else {"X-RateLimit-Remaining": remaining}
    return urllib.error.HTTPError(
        "https://api.github.com/x", code, "error", headers, None
    )


def install_urlopen(monkeypatch, behavior) -> None:
    """Replace urlopen with a function or raise the provided error."""

    def fake_urlopen(request, timeout):
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior

    monkeypatch.setattr(
        "topoptcomec.update.transport.urllib.request.urlopen", fake_urlopen
    )


def test_get_json_returns_status_headers_and_body(monkeypatch):
    """Successful responses are returned unchanged."""
    install_urlopen(monkeypatch, FakeResponse(b'{"ok": true}'))
    status, headers, body = UrllibTransport().get_json(
        "https://api.github.com/x", timeout=5
    )
    assert status == 200
    assert body == b'{"ok": true}'
    assert headers["Content-Type"] == "application/json"


def test_rate_limit_header_maps_to_rate_limit_error(monkeypatch):
    """An exhausted GitHub quota has a dedicated error."""
    install_urlopen(monkeypatch, http_error(403, remaining="0"))
    with pytest.raises(RateLimitError, match="rate limit"):
        UrllibTransport().get_json("https://api.github.com/x", timeout=5)


def test_plain_forbidden_maps_to_network_error(monkeypatch):
    """A 403 without an exhausted quota is a generic request failure."""
    install_urlopen(monkeypatch, http_error(403, remaining="12"))
    with pytest.raises(NetworkError):
        UrllibTransport().get_json("https://api.github.com/x", timeout=5)


def test_not_found_maps_to_release_metadata_error(monkeypatch):
    """A missing release is reported as absent metadata."""
    install_urlopen(monkeypatch, http_error(404))
    with pytest.raises(ReleaseMetadataError, match="No stable release"):
        UrllibTransport().get_json("https://api.github.com/x", timeout=5)


def test_server_error_maps_to_network_error(monkeypatch):
    """GitHub server failures are network errors."""
    install_urlopen(monkeypatch, http_error(500))
    with pytest.raises(NetworkError, match="HTTP 500"):
        UrllibTransport().get_json("https://api.github.com/x", timeout=5)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        TimeoutError("timed out"),
    ],
)
def test_connection_failures_map_to_network_error(monkeypatch, error):
    """Offline operation and timeouts raise a typed network error."""
    install_urlopen(monkeypatch, error)
    with pytest.raises(NetworkError):
        UrllibTransport().get_json("https://api.github.com/x", timeout=5)


def test_open_stream_returns_download_response(monkeypatch):
    """Downloads use the same transport and expose the response stream."""
    response = FakeResponse(b"artifact")
    install_urlopen(monkeypatch, response)

    stream = UrllibTransport().open_stream("https://github.com/x.zip", timeout=5)

    assert stream.read() == b"artifact"
    stream.close()
    assert response.closed


def test_open_stream_maps_download_errors(monkeypatch):
    """Download HTTP errors become typed network errors."""
    install_urlopen(monkeypatch, urllib.error.URLError("offline"))
    with pytest.raises(NetworkError):
        UrllibTransport().open_stream("https://github.com/x.zip", timeout=5)


# --- Verified downloads and safe installation ---


class FakeStream:
    """Stream that can fail after a number of reads."""

    def __init__(self, chunks: list[bytes], error_after: int | None = None) -> None:
        self._chunks = chunks
        self._index = 0
        self._error_after = error_after
        self.was_closed = False

    def read(self, size: int = -1) -> bytes:
        if self._error_after is not None and self._index >= self._error_after:
            raise OSError("connection reset")
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    def close(self) -> None:
        self.was_closed = True


class FakeDownloadTransport:
    """Download transport backed by in-memory chunks."""

    def __init__(self, chunks: list[bytes], error_after: int | None = None) -> None:
        self.chunks = chunks
        self.error_after = error_after
        self.opened: list[tuple[str, float]] = []
        self.stream: FakeStream | None = None

    def get_json(self, url, timeout, headers=None):  # pragma: no cover - unused
        raise AssertionError("metadata requests are not expected")

    def open_stream(self, url: str, timeout: float) -> FakeStream:
        self.opened.append((url, timeout))
        self.stream = FakeStream(self.chunks, self.error_after)
        return self.stream


def make_installer_check(
    installed="1.0.0",
    latest="2.0.0",
    tag="v2.0.0",
    status=UpdateStatus.UPDATE_AVAILABLE,
) -> UpdateCheck:
    release = ReleaseInfo(
        tag=tag,
        version=Version(latest),
        name=f"TopoptComec {latest}",
        notes="notes",
        page_url=f"https://github.com/ninja7v/TopoptComec/releases/tag/{tag}",
        assets=(),
    )
    return UpdateCheck(
        installed_version=Version(installed),
        latest_version=Version(latest),
        status=status,
        release=release,
    )


def make_asset(content: bytes, *, digest: str | None = None, size: int | None = None):
    return ReleaseAsset(
        name="TopoptComec-update.zip",
        url=(
            "https://github.com/ninja7v/TopoptComec/releases/download/"
            "v2.0.0/TopoptComec-update.zip"
        ),
        size=len(content) if size is None else size,
        sha256=digest,
    )


# --- Downloads -----------------------------------------------------------


def test_download_verifies_and_stores_artifact(tmp_path):
    """A verified download is moved into place atomically."""
    content = b"verified update payload"
    digest = hashlib.sha256(content).hexdigest()
    progress: list[tuple[int, str]] = []
    transport = FakeDownloadTransport([content[:8], content[8:]])

    target = download_asset(
        make_asset(content, digest=digest),
        tmp_path / "staging",
        transport=transport,
        progress=lambda percent, message: progress.append((percent, message)),
    )

    assert target.read_bytes() == content
    assert not (tmp_path / "staging" / "TopoptComec-update.zip.part").exists()
    assert progress[-1][0] == 100
    assert target.parent == tmp_path / "staging"
    assert transport.stream is not None and transport.stream.was_closed


def test_download_rejects_checksum_mismatch_and_keeps_old_file(tmp_path):
    """A corrupt download never replaces an existing artifact."""
    staging = tmp_path / "staging"
    staging.mkdir()
    existing = staging / "TopoptComec-update.zip"
    existing.write_bytes(b"old working version")
    content = b"tampered payload"
    transport = FakeDownloadTransport([content])

    with pytest.raises(ChecksumError):
        download_asset(
            make_asset(content, digest="0" * 64),
            staging,
            transport=transport,
        )

    assert existing.read_bytes() == b"old working version"
    assert not (staging / "TopoptComec-update.zip.part").exists()


def test_download_requires_published_digest(tmp_path):
    """Artifacts without a published digest are not treated as verified."""
    with pytest.raises(ChecksumError):
        download_asset(
            make_asset(b"payload", digest=None),
            tmp_path,
            transport=FakeDownloadTransport([b"payload"]),
        )


def test_download_handles_interrupted_transfer(tmp_path):
    """An interrupted transfer is reported and leaves no partial file."""
    content = b"interrupted payload"
    digest = hashlib.sha256(content).hexdigest()
    transport = FakeDownloadTransport([content[:5]], error_after=1)

    with pytest.raises(DownloadError):
        download_asset(
            make_asset(content, digest=digest),
            tmp_path,
            transport=transport,
        )

    assert list(tmp_path.iterdir()) == []


def test_download_detects_incomplete_file(tmp_path):
    """A short download that matches the digest length but not size fails."""
    content = b"short"
    digest = hashlib.sha256(content).hexdigest()
    with pytest.raises(DownloadError):
        download_asset(
            make_asset(content, digest=digest, size=999),
            tmp_path,
            transport=FakeDownloadTransport([content]),
        )
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_untrusted_url(tmp_path):
    """Non-HTTPS or foreign hosts are refused before any request."""
    asset = ReleaseAsset(
        name="update.zip",
        url="https://evil.example/update.zip",
        size=1,
        sha256="a" * 64,
    )
    transport = FakeDownloadTransport([b"x"])
    with pytest.raises(ReleaseMetadataError, match="not trusted"):
        download_asset(asset, tmp_path, transport=transport)
    assert transport.opened == []


def test_download_rejects_untrusted_redirect(tmp_path):
    """A redirect away from trusted hosts is refused before reading."""
    content = b"payload"

    class RedirectingTransport(FakeDownloadTransport):
        def open_stream(self, url: str, timeout: float) -> FakeStream:
            stream = super().open_stream(url, timeout)
            stream.geturl = lambda: "https://evil.example/update.zip"  # type: ignore[attr-defined]
            return stream

    with pytest.raises(ReleaseMetadataError, match="not trusted"):
        download_asset(
            make_asset(content, digest=hashlib.sha256(content).hexdigest()),
            tmp_path,
            transport=RedirectingTransport([content]),
        )
    assert list(tmp_path.iterdir()) == []


def test_download_does_not_touch_user_data(tmp_path):
    """Only the staging directory is written."""
    user_dir = tmp_path / "user"
    results = user_dir / "results" / "Bridge"
    results.mkdir(parents=True)
    result_file = results / "Bridge.png"
    result_file.write_bytes(b"precious result")
    presets = user_dir / "presets.json"
    presets.write_text('{"Mine": {}}', encoding="utf-8")

    content = b"payload"
    download_asset(
        make_asset(content, digest=hashlib.sha256(content).hexdigest()),
        user_dir / "updates",
        transport=FakeDownloadTransport([content]),
    )

    assert result_file.read_bytes() == b"precious result"
    assert presets.read_text(encoding="utf-8") == '{"Mine": {}}'


# --- Installation plans --------------------------------------------------


def test_detect_frozen_installation_is_not_automatically_updatable():
    """Standalone executables receive manual instructions."""
    installation = detect_installation(frozen=True)
    assert installation.kind is InstallationKind.FROZEN

    plan = build_plan(make_installer_check(), installation)
    assert not plan.supported
    assert "release page" in plan.instructions.lower()
    assert "releases/tag/v2.0.0" in plan.instructions


def test_detect_non_git_installation_is_not_automatically_updatable(tmp_path):
    """A site-packages style installation is not replaced automatically."""
    installation = detect_installation(tmp_path, frozen=False)
    assert installation.kind in {
        InstallationKind.PACKAGE,
        InstallationKind.UNKNOWN,
    }
    plan = build_plan(make_installer_check(), installation)
    assert not plan.supported
    assert plan.instructions


def test_git_installation_plan_is_supported(tmp_path):
    """A git checkout is updated in place with a fast-forward."""
    installation = Installation(InstallationKind.GIT_CHECKOUT, tmp_path, "Git")
    plan = build_plan(make_installer_check(), installation)
    assert plan.supported
    assert "git" in plan.instructions
    assert "merge --ff-only" in plan.instructions


def test_install_refuses_unsupported_installations(tmp_path):
    """Frozen/package installs are never modified by the updater."""
    installation = Installation(InstallationKind.FROZEN, tmp_path / "app.exe")
    with pytest.raises(InstallationError, match="release page"):
        install_update(
            make_installer_check(), build_plan(make_installer_check(), installation)
        )


def test_install_is_a_no_op_when_already_current(tmp_path):
    """No installation is attempted when the check found no update."""
    check = make_installer_check(installed="2.0.0", status=UpdateStatus.UP_TO_DATE)
    result = install_update(
        check,
        build_plan(check, Installation(InstallationKind.GIT_CHECKOUT, tmp_path)),
    )
    assert result.success
    assert not result.restart_required


# --- Git updates ---------------------------------------------------------


def git(cwd: Path, *arguments: str) -> str:
    """Run git in a test repository and return stdout."""
    completed = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def make_remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote and a clone at tag v1.0.0."""
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    work = tmp_path / "work"
    subprocess.run(
        ["git", "clone", str(remote), str(work)],
        check=True,
        capture_output=True,
        text=True,
    )
    git(work, "config", "user.email", "tests@example.com")
    git(work, "config", "user.name", "Tests")
    (work / "app.py").write_text("VERSION = '1.0.0'\n", encoding="utf-8")
    git(work, "add", "app.py")
    git(work, "commit", "-m", "initial")
    git(work, "tag", "v1.0.0")
    git(work, "push", "origin", "main", "--tags")
    return remote, work


def publish_release(work: Path, tmp_path: Path, version: str) -> None:
    """Publish a new commit and tag from a second clone."""
    publisher = tmp_path / f"publisher-{version}"
    subprocess.run(
        ["git", "clone", str(work.parent / "remote.git"), str(publisher)],
        check=True,
        capture_output=True,
        text=True,
    )
    git(publisher, "config", "user.email", "tests@example.com")
    git(publisher, "config", "user.name", "Tests")
    (publisher / "app.py").write_text(f"VERSION = '{version}'\n", encoding="utf-8")
    git(publisher, "add", "app.py")
    git(publisher, "commit", "-m", f"release {version}")
    git(publisher, "tag", f"v{version}")
    git(publisher, "push", "origin", "main", "--tags")


def git_installation(work: Path) -> Installation:
    return Installation(InstallationKind.GIT_CHECKOUT, work, "Git checkout")


def test_git_update_fast_forwards_clean_checkout(tmp_path):
    """A clean, behind checkout is fast-forwarded to the release tag."""
    remote, work = make_remote_and_clone(tmp_path)
    publish_release(work, tmp_path, "2.0.0")
    check = make_installer_check()
    stages: list[str] = []

    result = install_update(
        check,
        build_plan(check, git_installation(work)),
        progress=lambda percent, message: stages.append(message),
    )

    assert result.success
    assert result.restart_required
    assert "2.0.0" in result.message
    assert (work / "app.py").read_text(encoding="utf-8") == "VERSION = '2.0.0'\n"
    assert git(work, "rev-parse", "HEAD") == git(remote, "rev-parse", "v2.0.0^{commit}")
    assert stages[-1] == "Update installed."


def test_git_update_refuses_dirty_checkout(tmp_path):
    """Uncommitted changes block the update and stay untouched."""
    _remote, work = make_remote_and_clone(tmp_path)
    publish_release(work, tmp_path, "2.0.0")
    (work / "app.py").write_text("local experiment\n", encoding="utf-8")
    before = git(work, "rev-parse", "HEAD")
    check = make_installer_check()

    with pytest.raises(InstallationError, match="uncommitted changes"):
        install_update(check, build_plan(check, git_installation(work)))

    assert (work / "app.py").read_text(encoding="utf-8") == "local experiment\n"
    assert git(work, "rev-parse", "HEAD") == before


def test_git_update_refuses_unrelated_origin(tmp_path):
    """Only checkouts tracking the official repository are updated."""
    _remote, work = make_remote_and_clone(tmp_path)
    git(work, "remote", "set-url", "origin", "https://github.com/other/fork.git")
    before = git(work, "rev-parse", "HEAD")
    check = make_installer_check()

    with pytest.raises(InstallationError, match="not the official"):
        install_update(check, build_plan(check, git_installation(work)))

    assert git(work, "rev-parse", "HEAD") == before


def test_git_update_refuses_checkout_without_origin(tmp_path):
    """A checkout without a remote is not guessed at."""
    _remote, work = make_remote_and_clone(tmp_path)
    git(work, "remote", "remove", "origin")
    before = git(work, "rev-parse", "HEAD")
    check = make_installer_check()

    with pytest.raises(InstallationError, match="no 'origin'"):
        install_update(check, build_plan(check, git_installation(work)))

    assert git(work, "rev-parse", "HEAD") == before


def test_git_update_refuses_diverged_history_without_changes(tmp_path):
    """Local commits that diverged from the release are not overwritten."""
    _remote, work = make_remote_and_clone(tmp_path)
    publish_release(work, tmp_path, "2.0.0")
    (work / "local.txt").write_text("local work\n", encoding="utf-8")
    git(work, "add", "local.txt")
    git(work, "commit", "-m", "local work")
    before = git(work, "rev-parse", "HEAD")
    check = make_installer_check()

    with pytest.raises(InstallationError, match="fast-forward"):
        install_update(check, build_plan(check, git_installation(work)))

    assert git(work, "rev-parse", "HEAD") == before
    assert (work / "local.txt").read_text(encoding="utf-8") == "local work\n"


def test_git_update_rejects_missing_release_tag(tmp_path):
    """A release tag absent from the remote is not guessed or installed."""
    _remote, work = make_remote_and_clone(tmp_path)
    before = git(work, "rev-parse", "HEAD")
    check = make_installer_check()

    with pytest.raises(InstallationError, match="was not found"):
        install_update(check, build_plan(check, git_installation(work)))

    assert git(work, "rev-parse", "HEAD") == before


def test_git_update_refuses_when_checkout_is_already_newer(tmp_path):
    """A stale version string cannot make the updater report a false success."""
    _remote, work = make_remote_and_clone(tmp_path)
    (work / "ahead.txt").write_text("newer code\n", encoding="utf-8")
    git(work, "add", "ahead.txt")
    git(work, "commit", "-m", "newer than the tag")
    before = git(work, "rev-parse", "HEAD")
    check = make_installer_check(installed="0.9.0", latest="1.0.0", tag="v1.0.0")

    with pytest.raises(InstallationError, match="different revision"):
        install_update(check, build_plan(check, git_installation(work)))

    assert git(work, "rev-parse", "HEAD") == before


def test_git_update_rejects_tag_version_mismatch(tmp_path):
    """Metadata whose tag and version disagree is refused."""
    _remote, work = make_remote_and_clone(tmp_path)
    publish_release(work, tmp_path, "2.0.0")
    before = git(work, "rev-parse", "HEAD")
    check = make_installer_check(latest="3.0", tag="v2.0.0")

    with pytest.raises(InstallationError, match="does not match"):
        install_update(check, build_plan(check, git_installation(work)))

    assert git(work, "rev-parse", "HEAD") == before


# --- Update command ---


def make_cli_check(
    installed: str = "3.0",
    latest: str = "3.1",
    status: UpdateStatus = UpdateStatus.UPDATE_AVAILABLE,
) -> UpdateCheck:
    release = ReleaseInfo(
        tag=f"v{latest}",
        version=Version(latest),
        name=f"TopoptComec {latest}",
        notes="- New feature\n- Bug fixes",
        page_url=f"https://github.com/ninja7v/TopoptComec/releases/tag/v{latest}",
    )
    return UpdateCheck(
        installed_version=Version(installed),
        latest_version=Version(latest),
        status=status,
        release=release,
    )


def git_plan(supported: bool = True) -> UpdatePlan:
    return UpdatePlan(
        installation=Installation(InstallationKind.GIT_CHECKOUT, Path("/repo"), "Git"),
        supported=supported,
        summary="Fast-forward the git checkout to v3.1.",
        instructions="git -C /repo merge --ff-only v3.1",
    )


def test_version_flag_reports_authoritative_version(capsys):
    """--version uses the same version source as the updater."""
    with (
        patch.object(sys, "argv", ["topoptcomec", "--version"]),
        pytest.raises(SystemExit) as exit_info,
    ):
        run_cli()
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"TopoptComec {__version__}"


def test_update_reports_up_to_date_without_installing(capsys):
    """No download or install happens when already current."""
    with (
        patch(
            "topoptcomec.cli.update.check_for_update",
            return_value=make_cli_check(status=UpdateStatus.UP_TO_DATE),
        ),
        patch("topoptcomec.cli.update.install_update") as install,
    ):
        code = run_update_command(["--check"])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert f"Installed version: {__version__}" in out
    assert "is up to date" in out
    install.assert_not_called()


def test_update_reports_installed_version_ahead(capsys):
    """A development version newer than the release is not downgraded."""
    with (
        patch(
            "topoptcomec.cli.update.check_for_update",
            return_value=make_cli_check(
                installed="3.2", latest="3.1", status=UpdateStatus.INSTALLED_AHEAD
            ),
        ),
        patch("topoptcomec.cli.update.install_update") as install,
    ):
        code = run_update_command([])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "newer than the latest release" in out
    install.assert_not_called()


def test_update_check_displays_versions_notes_and_plan(capsys):
    """--check shows the available version, notes, and installation method."""
    with (
        patch("topoptcomec.cli.update.check_for_update", return_value=make_cli_check()),
        patch(
            "topoptcomec.cli.update.detect_installation",
            return_value=git_plan().installation,
        ),
        patch("topoptcomec.cli.update.build_plan", return_value=git_plan()),
        patch("topoptcomec.cli.update.install_update") as install,
    ):
        code = run_update_command(["--check"])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Update available: 3.1" in out
    assert "TopoptComec 3.1" in out
    assert "New feature" in out
    assert "releases/tag/v3.1" in out
    assert "Installation method: Git" in out
    assert "Fast-forward the git checkout" in out
    assert "not installed" in out
    install.assert_not_called()


def test_update_install_with_yes_is_non_interactive(capsys):
    """--yes installs without prompting and reports the restart."""
    with (
        patch("topoptcomec.cli.update.check_for_update", return_value=make_cli_check()),
        patch(
            "topoptcomec.cli.update.detect_installation",
            return_value=git_plan().installation,
        ),
        patch("topoptcomec.cli.update.build_plan", return_value=git_plan()),
        patch(
            "topoptcomec.cli.update.install_update",
            return_value=UpdateResult(
                success=True,
                restart_required=True,
                message="TopoptComec was updated to 3.1.",
            ),
        ) as install,
        patch("builtins.input") as ask,
    ):
        code = run_update_command(["--yes"])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "TopoptComec was updated to 3.1." in out
    assert "Restart TopoptComec" in out
    ask.assert_not_called()
    install.assert_called_once()
    assert callable(install.call_args.kwargs["progress"])


def test_update_prompts_and_respects_declined_confirmation(capsys):
    """Declining the prompt leaves the installation unchanged."""
    with (
        patch("topoptcomec.cli.update.check_for_update", return_value=make_cli_check()),
        patch(
            "topoptcomec.cli.update.detect_installation",
            return_value=git_plan().installation,
        ),
        patch("topoptcomec.cli.update.build_plan", return_value=git_plan()),
        patch("topoptcomec.cli.update._stdin_is_interactive", return_value=True),
        patch("builtins.input", return_value="n") as ask,
        patch("topoptcomec.cli.update.install_update") as install,
    ):
        code = run_update_command([])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Update cancelled" in out
    ask.assert_called_once()
    install.assert_not_called()


def test_update_requires_yes_in_non_interactive_session(capsys):
    """A pipe cannot confirm an update implicitly."""
    with (
        patch("topoptcomec.cli.update.check_for_update", return_value=make_cli_check()),
        patch(
            "topoptcomec.cli.update.detect_installation",
            return_value=git_plan().installation,
        ),
        patch("topoptcomec.cli.update.build_plan", return_value=git_plan()),
        patch("topoptcomec.cli.update._stdin_is_interactive", return_value=False),
        patch("topoptcomec.cli.update.install_update") as install,
    ):
        code = run_update_command([])

    captured = capsys.readouterr()
    assert code == EXIT_CONFIRMATION_REQUIRED
    assert "--yes" in captured.err
    install.assert_not_called()


def test_update_returns_manual_exit_code_for_unsupported_install(capsys):
    """Unsupported installation methods get instructions and exit code 3."""
    plan = UpdatePlan(
        installation=Installation(
            InstallationKind.FROZEN, Path("/app"), "Standalone executable"
        ),
        supported=False,
        summary="Standalone executables cannot be replaced automatically.",
        instructions="Download the release from https://github.com/ninja7v/TopoptComec",
    )
    with (
        patch("topoptcomec.cli.update.check_for_update", return_value=make_cli_check()),
        patch(
            "topoptcomec.cli.update.detect_installation",
            return_value=plan.installation,
        ),
        patch("topoptcomec.cli.update.build_plan", return_value=plan),
        patch("topoptcomec.cli.update.install_update") as install,
    ):
        code = run_update_command(["--yes"])

    captured = capsys.readouterr()
    assert code == EXIT_MANUAL_ACTION
    assert "not supported" in captured.err
    assert "Download the release" in captured.out
    install.assert_not_called()


def test_update_network_error_returns_error_exit_code(capsys):
    """Offline operation fails cleanly without touching the installation."""
    with patch(
        "topoptcomec.cli.update.check_for_update",
        side_effect=NetworkError("connection refused"),
    ):
        code = run_update_command([])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "could not check for updates" in captured.err
    assert "connection refused" in captured.err


def test_update_rate_limit_has_a_clear_message(capsys):
    """GitHub API rate limiting is explained to the user."""
    with patch(
        "topoptcomec.cli.update.check_for_update",
        side_effect=RateLimitError("rate limit exceeded"),
    ):
        code = run_update_command([])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "rate limit exceeded" in captured.err


def test_update_installation_failure_returns_error_and_recovery(capsys):
    """A failed install reports recovery instructions."""
    with (
        patch("topoptcomec.cli.update.check_for_update", return_value=make_cli_check()),
        patch(
            "topoptcomec.cli.update.detect_installation",
            return_value=git_plan().installation,
        ),
        patch("topoptcomec.cli.update.build_plan", return_value=git_plan()),
        patch(
            "topoptcomec.cli.update.install_update",
            side_effect=InstallationError("working tree is dirty"),
        ),
    ):
        code = run_update_command(["--yes"])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "working tree is dirty" in captured.err
    assert "left in place" in captured.err
    assert "releases/tag/v3.1" in captured.err


def test_update_rejects_invalid_timeout(capsys):
    """A non-positive timeout is a usage error."""
    code = run_update_command(["--timeout", "0"])
    assert code == EXIT_ERROR
    assert "--timeout" in capsys.readouterr().err


def test_main_dispatches_update_without_starting_gui():
    """``topoptcomec update`` is routed to the update command."""
    from topoptcomec.__main__ import main

    with (
        patch.object(sys, "argv", ["topoptcomec", "update", "--check"]),
        patch(
            "topoptcomec.cli.update.run_update_command", return_value=EXIT_OK
        ) as command,
        patch("PySide6.QtWidgets.QApplication") as application,
        pytest.raises(SystemExit) as exit_info,
    ):
        main()

    assert exit_info.value.code == EXIT_OK
    command.assert_called_once_with(["--check"])
    application.assert_not_called()


# --- GUI update flow ---


def make_gui_check(
    installed="3.0",
    latest="3.1",
    status=UpdateStatus.UPDATE_AVAILABLE,
) -> UpdateCheck:
    return UpdateCheck(
        installed_version=Version(installed),
        latest_version=Version(latest),
        status=status,
        release=ReleaseInfo(
            tag=f"v{latest}",
            version=Version(latest),
            name=f"TopoptComec {latest}",
            notes="notes",
            page_url=f"https://github.com/ninja7v/TopoptComec/releases/tag/v{latest}",
        ),
    )


def make_plan(supported: bool = True) -> UpdatePlan:
    return UpdatePlan(
        installation=Installation(InstallationKind.GIT_CHECKOUT, Path("/repo"), "Git"),
        supported=supported,
        summary="Fast-forward the git checkout.",
        instructions="git -C /repo merge --ff-only v3.1",
    )


def wait_until(qt_app, predicate, timeout_ms: int = 5000) -> None:
    """Process Qt events until predicate is true or the timeout expires."""
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        qt_app.processEvents()
        if predicate():
            return
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the GUI state")
        QTest.qWait(5)


def trigger_check(window: MainWindow, qt_app) -> None:
    """Click the update button and wait for the background check to finish."""
    window.header.update_button.click()
    wait_until(qt_app, lambda: window.update_check_worker is None)


def test_up_to_date_shows_small_message(qt_app):
    """No update panel: only a short message is shown."""
    window = MainWindow()
    try:
        check = make_gui_check(
            installed="3.1", latest="3.1", status=UpdateStatus.UP_TO_DATE
        )
        with (
            patch("topoptcomec.gui.workers.check_for_update", return_value=check),
            patch("topoptcomec.gui.workers.build_plan", return_value=make_plan()),
            patch.object(QMessageBox, "information") as information,
        ):
            trigger_check(window, qt_app)

        information.assert_called_once()
        assert (
            information.call_args.args[2]
            == "You already run the latest version (v3.1)."
        )
        assert window.header.update_button.isEnabled()
    finally:
        window.close()


def test_update_available_asks_yes_no_and_installs(qt_app):
    """An available update asks one simple Yes/No question."""
    window = MainWindow()
    try:
        check = make_gui_check()
        with (
            patch("topoptcomec.gui.workers.check_for_update", return_value=check),
            patch("topoptcomec.gui.workers.build_plan", return_value=make_plan()),
            patch.object(
                QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
            ) as question,
            patch.object(window, "_install_update") as install,
        ):
            trigger_check(window, qt_app)

        question.assert_called_once()
        assert question.call_args.args[2] == (
            "Would you like to update from your (v3.0) version to the latest "
            "version (v3.1)?"
        )
        install.assert_called_once_with(check, make_plan())
    finally:
        window.close()


def test_update_available_declined_changes_nothing(qt_app):
    """Answering No leaves the installation alone."""
    window = MainWindow()
    try:
        with (
            patch(
                "topoptcomec.gui.workers.check_for_update",
                return_value=make_gui_check(),
            ),
            patch("topoptcomec.gui.workers.build_plan", return_value=make_plan()),
            patch.object(
                QMessageBox, "question", return_value=QMessageBox.StandardButton.No
            ),
            patch.object(window, "_install_update") as install,
        ):
            trigger_check(window, qt_app)

        install.assert_not_called()
        assert window.status_bar.currentMessage() == "Update cancelled."
        assert window.header.update_button.isEnabled()
    finally:
        window.close()


def test_unsupported_install_shows_manual_instructions(qt_app):
    """Installations that cannot self-update get instructions, not a prompt."""
    window = MainWindow()
    try:
        plan = make_plan(supported=False)
        with (
            patch(
                "topoptcomec.gui.workers.check_for_update",
                return_value=make_gui_check(),
            ),
            patch("topoptcomec.gui.workers.build_plan", return_value=plan),
            patch.object(QMessageBox, "information") as information,
            patch.object(window, "_install_update") as install,
        ):
            trigger_check(window, qt_app)

        text = information.call_args.args[2]
        assert "must be installed manually" in text
        assert plan.instructions in text
        install.assert_not_called()
    finally:
        window.close()


def test_check_failure_is_reported_and_controls_recover(qt_app):
    """Network failures show a warning and re-enable the update button."""
    window = MainWindow()
    try:
        with (
            patch(
                "topoptcomec.gui.workers.check_for_update",
                side_effect=NetworkError("offline"),
            ),
            patch.object(QMessageBox, "warning") as warning,
        ):
            trigger_check(window, qt_app)

        warning.assert_called_once()
        assert "offline" in warning.call_args.args[2]
        assert window.header.update_button.isEnabled()
    finally:
        window.close()


def test_check_runs_in_background_and_ignores_duplicates(qt_app):
    """Clicking update never freezes the GUI; repeated clicks are ignored."""
    gate = threading.Event()
    started = threading.Event()
    calls: list[int] = []

    def blocking_check():
        calls.append(1)
        started.set()
        if not gate.wait(5):
            raise NetworkError("test gate timed out")
        return make_gui_check(
            installed="3.1", latest="3.1", status=UpdateStatus.UP_TO_DATE
        )

    window = MainWindow()
    try:
        with (
            patch("topoptcomec.gui.workers.check_for_update", blocking_check),
            patch("topoptcomec.gui.workers.build_plan", return_value=make_plan()),
            patch.object(QMessageBox, "information"),
        ):
            begin = time.monotonic()
            window.header.update_button.click()
            assert time.monotonic() - begin < 0.5

            assert started.wait(5)
            assert not window.header.update_button.isEnabled()
            window.header.update_button.click()  # disabled button: no new check
            window._check_for_updates()  # direct duplicate call: ignored
            assert len(calls) == 1

            gate.set()
            wait_until(qt_app, lambda: window.update_check_worker is None)

        assert window.header.update_button.isEnabled()
    finally:
        window.close()


def test_install_reports_progress_and_restart(qt_app):
    """Progress is shown in the status bar and the restart is announced."""

    def fake_install(check, plan, progress):
        progress(40, "Downloading update...")
        progress(100, "Update installed.")
        return UpdateResult(
            success=True,
            restart_required=True,
            message="TopoptComec was updated to 3.1.",
        )

    window = MainWindow()
    try:
        check = make_gui_check()
        window._last_update_check = check
        with (
            patch("topoptcomec.gui.workers.install_update", fake_install),
            patch.object(QMessageBox, "information") as information,
        ):
            window._install_update(check, make_plan())
            wait_until(qt_app, lambda: window.update_install_worker is None)

        message = information.call_args.args[2]
        assert "TopoptComec was updated to 3.1." in message
        assert "Restart TopoptComec" in message
        assert window.progress_bar.isHidden()
        assert window.header.update_button.isEnabled()
    finally:
        window.close()


def test_install_failure_reports_recovery_and_keeps_controls(qt_app):
    """A failed installation keeps the app usable and shows the release page."""

    def failing_install(check, plan, progress):
        raise InstallationError("the checkout has uncommitted changes")

    window = MainWindow()
    try:
        check = make_gui_check()
        window._last_update_check = check
        with (
            patch("topoptcomec.gui.workers.install_update", failing_install),
            patch.object(QMessageBox, "critical") as critical,
        ):
            window._install_update(check, make_plan())
            wait_until(qt_app, lambda: window.update_install_worker is None)

        text = critical.call_args.args[2]
        assert "uncommitted changes" in text
        assert "left unchanged" in text
        assert check.release.page_url in text
        assert window.header.update_button.isEnabled()
    finally:
        window.close()


def test_clicking_title_opens_github(qt_app):
    """The title replaces the former info button."""
    window = MainWindow()
    try:
        with patch("topoptcomec.gui.main_window.QDesktopServices.openUrl") as open_url:
            window.header.title_label.click()

        open_url.assert_called_once()
        assert open_url.call_args.args[0].toString() == (
            "https://github.com/ninja7v/TopoptComec"
        )
    finally:
        window.close()
