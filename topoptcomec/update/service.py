# topoptcomec/update/service.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Installation-independent update checking service.

from __future__ import annotations

from packaging.version import Version

from ..version import __version__
from .github import GithubReleaseClient, parse_version_tag
from .models import ReleaseInfo, UpdateCheck, UpdateStatus


def determine_status(installed: Version, latest: Version) -> UpdateStatus:
    """Compare two versions and return the update status.

    Prereleases are never offered as updates, even if they compare higher.

    Parameters
    ----------
    installed : Version
        Version of the running application.
    latest : Version
        Version of the latest published release.

    Returns
    -------
    UpdateStatus
        Whether an update is available, the app is current, or the installed
        version is ahead of the published release.
    """
    if latest.is_prerelease:
        return UpdateStatus.UP_TO_DATE
    if installed < latest:
        return UpdateStatus.UPDATE_AVAILABLE
    if installed > latest:
        return UpdateStatus.INSTALLED_AHEAD
    return UpdateStatus.UP_TO_DATE


def check_for_update(
    installed_version: str | Version | None = None,
    *,
    client: GithubReleaseClient | None = None,
) -> UpdateCheck:
    """Check GitHub for a newer stable release.

    Parameters
    ----------
    installed_version : str | Version | None
        Version to compare against; defaults to the running application
        version. A leading ``v`` is accepted.
    client : GithubReleaseClient | None
        Client to use; a default client is created when omitted.

    Returns
    -------
    UpdateCheck
        Installed version, latest version, status, and release metadata.

    Raises
    ------
    VersionError
        If the installed version cannot be parsed.
    NetworkError
        If the release cannot be fetched.
    ReleaseMetadataError
        If the response is malformed or untrusted.
    """
    installed = (
        parse_version_tag(installed_version)
        if installed_version is not None
        else parse_version_tag(__version__)
    )
    release: ReleaseInfo = (client or GithubReleaseClient()).latest_release()
    return UpdateCheck(
        installed_version=installed,
        latest_version=release.version,
        status=determine_status(installed, release.version),
        release=release,
    )
