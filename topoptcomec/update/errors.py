# topoptcomec/update/errors.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Errors raised by the update service.

from __future__ import annotations


class UpdateError(Exception):
    """Base class for every recoverable update failure."""


class VersionError(UpdateError):
    """A version string could not be parsed."""


class NetworkError(UpdateError):
    """The update service could not reach the remote server."""


class RateLimitError(NetworkError):
    """GitHub rejected the request because the API rate limit was reached."""


class ReleaseMetadataError(UpdateError):
    """The release metadata is missing, malformed, or untrusted."""


class DownloadError(UpdateError):
    """An update artifact could not be downloaded."""


class ChecksumError(DownloadError):
    """A downloaded artifact failed or lacked an integrity check."""


class InstallationError(UpdateError):
    """An update could not be installed without risking user data."""
