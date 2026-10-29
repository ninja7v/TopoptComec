# topoptcomec/update/installer.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Installation detection, verified downloads, and safe update installation.
#
# The updater only ever acts on a clean git checkout. Frozen executables and
# packages installed into site-packages receive instructions instead of an
# automatic replacement, because the project does not publish installable
# binaries yet.

from __future__ import annotations

import hashlib
import http.client
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ..version import __version__
from .errors import (
    ChecksumError,
    DownloadError,
    InstallationError,
    UpdateError,
)
from .github import parse_version_tag, validate_download_url
from .models import (
    Installation,
    InstallationKind,
    ReleaseAsset,
    UpdateCheck,
    UpdatePlan,
    UpdateResult,
)
from .transport import Transport, UrllibTransport

#: Progress callback: (percent 0-100, human-readable stage).
ProgressCallback = Callable[[int, str], None]

#: Upper bound for a single update artifact, as a guard against hostile metadata.
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024

#: Directory (inside the user's home) where manually downloaded updates land.
UPDATE_STAGING_DIR = Path.home() / ".topoptcomec" / "updates"
_GIT_TIMEOUT = 120
_DOWNLOAD_CHUNK = 64 * 1024


def detect_installation(
    package_directory: Path | None = None, *, frozen: bool | None = None
) -> Installation:
    """Determine how the running TopoptComec was installed.

    Parameters
    ----------
    package_directory : Path | None
        Directory containing the ``topoptcomec`` package. Defaults to the
        location of the running package.
    frozen : bool | None
        Overrides PyInstaller detection (used by tests).

    Returns
    -------
    Installation
        Installation kind and location.
    """
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if is_frozen:
        return Installation(
            kind=InstallationKind.FROZEN,
            location=Path(sys.executable),
            detail="Standalone executable (PyInstaller)",
        )

    package_dir = package_directory or Path(__file__).resolve().parent.parent
    repository = _git_toplevel(package_dir)
    if repository is not None:
        return Installation(
            kind=InstallationKind.GIT_CHECKOUT,
            location=repository,
            detail="Git checkout",
        )

    if _distribution_installed():
        return Installation(
            kind=InstallationKind.PACKAGE,
            location=package_dir,
            detail="Installed Python package",
        )
    return Installation(
        kind=InstallationKind.UNKNOWN,
        location=package_dir,
        detail="Unrecognized installation",
    )


def build_plan(
    check: UpdateCheck, installation: Installation | None = None
) -> UpdatePlan:
    """Describe how an available update can be applied to this installation.

    Parameters
    ----------
    check : UpdateCheck
        Result of :func:`topoptcomec.update.check_for_update`.
    installation : Installation | None
        Installation to update; detected automatically when omitted.

    Returns
    -------
    UpdatePlan
        Whether automatic installation is supported, plus instructions.
    """
    installation = installation or detect_installation()
    tag = check.release.tag
    version = check.latest_version
    page = check.release.page_url

    if installation.kind is InstallationKind.GIT_CHECKOUT:
        location = installation.location or Path()
        return UpdatePlan(
            installation=installation,
            supported=True,
            summary=(
                f"Fast-forward the git checkout at {location} to {tag}. "
                "Uncommitted changes are detected and block the update."
            ),
            instructions=(
                f"Update manually with:\n"
                f"  git -C {location} fetch --tags origin\n"
                f"  git -C {location} merge --ff-only {tag}\n"
                f"If the checkout has local changes, commit or stash them first."
            ),
        )

    if installation.kind is InstallationKind.FROZEN:
        return UpdatePlan(
            installation=installation,
            supported=False,
            summary=(
                "This standalone executable cannot be replaced while it is "
                "running, and no verified platform binaries are published for "
                "automatic installation."
            ),
            instructions=(
                f"Download TopoptComec {version} from the release page:\n"
                f"  {page}\n"
                "Quit TopoptComec, replace the existing executable with the "
                "downloaded one, then start the new version."
            ),
        )

    if installation.kind is InstallationKind.PACKAGE:
        return UpdatePlan(
            installation=installation,
            supported=False,
            summary=(
                "TopoptComec is installed as a Python package; it is not "
                "published on a package index, so the updater cannot reinstall "
                "it automatically."
            ),
            instructions=(
                f"Install version {version} from the release page:\n"
                f"  {page}\n"
                "Either clone the repository at the release tag and run "
                "`uv sync`, or reinstall the package from the downloaded "
                "source archive."
            ),
        )

    return UpdatePlan(
        installation=installation,
        supported=False,
        summary="The installation method could not be recognized.",
        instructions=(
            f"Download version {version} manually from:\n  {page}\n"
            "Follow the installation instructions that match the way "
            "TopoptComec was installed."
        ),
    )


def install_update(
    check: UpdateCheck,
    plan: UpdatePlan | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> UpdateResult:
    """Install an available update.

    Only a clean git checkout is updated automatically. Every other
    installation raises :class:`InstallationError` carrying manual
    instructions, so the running installation is never modified blindly.

    Parameters
    ----------
    check : UpdateCheck
        Update check describing the target release.
    plan : UpdatePlan | None
        Plan from :func:`build_plan`; rebuilt when omitted.
    progress : ProgressCallback | None
        Called with a percentage and a stage description.

    Returns
    -------
    UpdateResult
        Success flag, restart requirement, and a user-facing message.

    Raises
    ------
    InstallationError
        If the installation cannot be updated safely.
    """
    if not check.update_available:
        return UpdateResult(
            success=True,
            restart_required=False,
            message=f"TopoptComec {__version__} is already up to date.",
        )

    plan = plan or build_plan(check)
    if not plan.supported:
        raise InstallationError(plan.instructions)
    if plan.installation.kind is not InstallationKind.GIT_CHECKOUT:
        raise InstallationError(plan.instructions)
    repository = plan.installation.location
    if repository is None:
        raise InstallationError(
            "The git checkout location is unknown; update manually from "
            f"{check.release.page_url}."
        )
    return _update_git_checkout(repository, check, progress)


def download_asset(
    asset: ReleaseAsset,
    destination_dir: Path,
    *,
    transport: Transport | None = None,
    timeout: float = 60.0,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download and verify a release asset.

    The file is written to a temporary ``.part`` file, hashed while it
    streams, and only moved into place after the SHA-256 digest published
    with the release matches. An artifact without a digest is never treated
    as verified.

    Parameters
    ----------
    asset : ReleaseAsset
        Asset metadata from the GitHub API.
    destination_dir : Path
        Directory in which to store the verified artifact.
    transport : Transport | None
        HTTP transport; the standard-library one is used when omitted.
    timeout : float
        Download timeout in seconds.
    progress : ProgressCallback | None
        Called with a percentage and a stage description.

    Returns
    -------
    Path
        Path to the verified artifact.

    Raises
    ------
    ChecksumError
        If no SHA-256 digest is published or the digest does not match.
    DownloadError
        If the download is interrupted, incomplete, or too large.
    ReleaseMetadataError
        If the download URL is not trusted.
    """
    validate_download_url(asset.url)
    if not asset.sha256:
        raise ChecksumError(
            f"No SHA-256 digest is published for {asset.name!r}; refusing to "
            "treat the download as verified."
        )
    if not asset.name or Path(asset.name).name != asset.name:
        raise DownloadError(f"Invalid update artifact name: {asset.name!r}")

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DownloadError(
            f"Could not create the download directory {destination_dir}: {error}"
        ) from error
    target = destination_dir / asset.name
    partial = destination_dir / f"{asset.name}.part"
    _remove_if_exists(partial)

    client = transport or UrllibTransport()
    digest = hashlib.sha256()
    downloaded = 0
    _report(progress, 0, f"Downloading {asset.name}...")
    try:
        stream = client.open_stream(asset.url, timeout)
        try:
            final_url = getattr(stream, "geturl", lambda: asset.url)()
            if final_url:
                validate_download_url(final_url)
            with open(partial, "wb") as handle:
                while True:
                    chunk = stream.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise DownloadError(
                            f"{asset.name!r} exceeds the maximum download size."
                        )
                    digest.update(chunk)
                    handle.write(chunk)
                    _report(
                        progress,
                        _percent(downloaded, asset.size),
                        f"Downloading {asset.name}...",
                    )
        finally:
            stream.close()
    except UpdateError:
        _remove_if_exists(partial)
        raise
    except (OSError, http.client.HTTPException) as error:
        _remove_if_exists(partial)
        raise DownloadError(f"Download of {asset.name!r} failed: {error}") from error

    if asset.size and downloaded != asset.size:
        _remove_if_exists(partial)
        raise DownloadError(
            f"Download of {asset.name!r} is incomplete "
            f"({downloaded} of {asset.size} bytes)."
        )
    if digest.hexdigest() != asset.sha256:
        _remove_if_exists(partial)
        raise ChecksumError(
            f"SHA-256 mismatch for {asset.name!r}; the update was discarded."
        )
    try:
        os.replace(partial, target)
    except OSError as error:
        _remove_if_exists(partial)
        raise DownloadError(
            f"Could not store the verified update {asset.name!r}: {error}"
        ) from error
    _report(progress, 100, f"Verified {asset.name}.")
    return target


def _update_git_checkout(
    repository: Path, check: UpdateCheck, progress: ProgressCallback | None
) -> UpdateResult:
    """Fast-forward a clean git checkout to the release tag."""
    _report(progress, 0, "Checking the working tree...")
    _verify_official_origin(repository)
    status = _run_git(repository, "status", "--porcelain")
    if status.stdout.strip():
        raise InstallationError(
            "The git checkout has uncommitted changes, so it was not updated. "
            "Commit or stash them and run the update again.\n"
            f"Changed files:\n{status.stdout.strip()}"
        )

    _report(progress, 15, "Fetching the release tag...")
    fetch = _run_git(repository, "fetch", "--tags", "--prune", "origin")
    if fetch.returncode != 0:
        raise InstallationError(
            "Could not fetch the release from the git remote: "
            f"{fetch.stderr.strip() or fetch.stdout.strip()}"
        )

    tag = check.release.tag
    _report(progress, 45, f"Locating release {tag}...")
    resolved = _run_git(
        repository, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"
    )
    if resolved.returncode != 0:
        raise InstallationError(
            f"Release tag {tag!r} was not found in the git remote. "
            "The checkout was not changed."
        )
    if parse_version_tag(tag) != check.latest_version:
        raise InstallationError(
            f"Release tag {tag!r} does not match version "
            f"{check.latest_version}; the update was refused."
        )

    _report(progress, 60, f"Applying update to {tag}...")
    merge = _run_git(repository, "merge", "--ff-only", "--no-edit", tag)
    if merge.returncode != 0:
        raise InstallationError(
            "The checkout cannot be fast-forwarded to the release "
            f"(local commits may have diverged). Nothing was changed.\n"
            f"{merge.stderr.strip() or merge.stdout.strip()}"
        )

    _report(progress, 90, "Verifying the new revision...")
    head = _run_git(repository, "rev-parse", "HEAD")
    if head.stdout.strip() != resolved.stdout.strip():
        raise InstallationError(
            "The checkout already contains a different revision than the "
            f"release tag {tag}; the installed version was not reported "
            "correctly and the update was refused."
        )
    _report(progress, 100, "Update installed.")
    revision = head.stdout.strip()[:7] or "unknown"
    return UpdateResult(
        success=True,
        restart_required=True,
        message=(
            f"TopoptComec was updated to {check.latest_version} "
            f"(commit {revision}). Restart TopoptComec to use the new version."
        ),
    )


def _verify_official_origin(repository: Path) -> None:
    """Refuse to update a checkout whose origin is not the official repository.

    Local filesystem remotes are accepted for development and tests; remote
    Git hosting URLs must point at ``ninja7v/TopoptComec``.
    """
    origin = _run_git(repository, "remote", "get-url", "origin")
    if origin.returncode != 0:
        raise InstallationError(
            "The git checkout has no 'origin' remote, so it cannot be updated."
        )
    url = origin.stdout.strip()
    lowered = url.lower().removesuffix("/").removesuffix(".git")
    if lowered.endswith(
        ("github.com:ninja7v/topoptcomec", "github.com/ninja7v/topoptcomec")
    ):
        return
    if "://" not in url and not url.startswith(("git@", "ssh:")):
        local_path = Path(url).expanduser()
        if local_path.exists():
            return
    raise InstallationError(
        f"The checkout's origin ({url}) is not the official TopoptComec "
        "repository; refusing to update it."
    )


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a git command inside the checkout and return its result."""
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except FileNotFoundError as error:
        raise InstallationError(
            "Git is not installed, so the checkout cannot be updated."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise InstallationError(
            f"Git command {' '.join(arguments)} timed out."
        ) from error


def _git_toplevel(directory: Path) -> Path | None:
    """Return the git checkout containing ``directory``, if any."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    top_level = Path(result.stdout.strip())
    return top_level if top_level.is_dir() else None


def _distribution_installed() -> bool:
    """Whether topoptcomec package metadata is importable for this interpreter."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("topoptcomec")
    except PackageNotFoundError:
        return False
    return True


def _percent(downloaded: int, total: int) -> int:
    """Return a bounded download percentage."""
    if total <= 0:
        return 0
    return min(99, int(downloaded * 100 / total))


def _report(progress: ProgressCallback | None, percent: int, message: str) -> None:
    """Invoke a progress callback when one was supplied."""
    if progress is not None:
        progress(percent, message)


def _remove_if_exists(path: Path) -> None:
    """Delete a partially downloaded file, ignoring a missing file."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


__all__ = [
    "MAX_DOWNLOAD_BYTES",
    "UPDATE_STAGING_DIR",
    "ProgressCallback",
    "build_plan",
    "detect_installation",
    "download_asset",
    "install_update",
]
