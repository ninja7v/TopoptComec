# topoptcomec/update/__init__.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Installation-independent update service shared by the CLI and the GUI.

from __future__ import annotations

from .errors import (
    ChecksumError,
    DownloadError,
    InstallationError,
    NetworkError,
    RateLimitError,
    ReleaseMetadataError,
    UpdateError,
    VersionError,
)
from .github import (
    LATEST_RELEASE_URL,
    REPOSITORY,
    GithubReleaseClient,
    parse_version_tag,
)
from .installer import (
    UPDATE_STAGING_DIR,
    build_plan,
    detect_installation,
    download_asset,
    install_update,
)
from .models import (
    Installation,
    InstallationKind,
    ReleaseAsset,
    ReleaseInfo,
    UpdateCheck,
    UpdatePlan,
    UpdateResult,
    UpdateStatus,
)
from .service import check_for_update, determine_status

__all__ = [
    "LATEST_RELEASE_URL",
    "REPOSITORY",
    "UPDATE_STAGING_DIR",
    "ChecksumError",
    "DownloadError",
    "GithubReleaseClient",
    "Installation",
    "InstallationError",
    "InstallationKind",
    "NetworkError",
    "RateLimitError",
    "ReleaseAsset",
    "ReleaseInfo",
    "ReleaseMetadataError",
    "UpdateCheck",
    "UpdateError",
    "UpdatePlan",
    "UpdateResult",
    "UpdateStatus",
    "VersionError",
    "build_plan",
    "check_for_update",
    "detect_installation",
    "determine_status",
    "download_asset",
    "install_update",
    "parse_version_tag",
]
