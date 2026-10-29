# topoptcomec/update/github.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# GitHub Releases API client and release metadata validation.

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from urllib.parse import quote, urlsplit

from packaging.version import InvalidVersion, Version

from ..version import __version__
from .errors import ReleaseMetadataError, VersionError
from .models import ReleaseAsset, ReleaseInfo
from .transport import Transport, UrllibTransport

#: GitHub repository used for update checks.
REPOSITORY = "ninja7v/TopoptComec"
#: Official endpoint for the latest stable release.
LATEST_RELEASE_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
#: Hosts allowed to serve release metadata and artifacts.
TRUSTED_HOSTS = frozenset(
    {"api.github.com", "github.com", "objects.githubusercontent.com"}
)
#: Only tags made of these characters are accepted (they are also passed to git).
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_DIGEST_PATTERN = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
_DEFAULT_TIMEOUT = 15.0


class GithubReleaseClient:
    """Fetches and validates the latest stable release of the repository."""

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        repository: str = REPOSITORY,
        api_url: str = LATEST_RELEASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the client.

        Parameters
        ----------
        transport : Transport | None
            HTTP transport; the standard-library one is used when omitted.
        repository : str
            Repository in ``owner/name`` form, used to validate URLs.
        api_url : str
            URL of the GitHub "latest release" endpoint.
        timeout : float
            Request timeout in seconds.
        """
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self._transport: Transport = transport or UrllibTransport()
        self.repository = repository
        self.api_url = api_url
        self.timeout = timeout

    def latest_release(self) -> ReleaseInfo:
        """Return the latest stable release, validated against expectations.

        Returns
        -------
        ReleaseInfo
            Parsed release metadata.

        Raises
        ------
        ReleaseMetadataError
            If the response is malformed, points to unexpected hosts, or
            describes a prerelease instead of a stable release.
        NetworkError
            If the request cannot be completed.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": f"TopoptComec/{__version__}",
        }
        status, _response_headers, body = self._transport.get_json(
            self.api_url, self.timeout, headers
        )
        if status != 200:
            raise ReleaseMetadataError(
                f"GitHub returned an unexpected HTTP status ({status})."
            )
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as error:
            raise ReleaseMetadataError(
                "GitHub returned a response that is not valid JSON."
            ) from error
        return parse_release(payload, repository=self.repository)


def parse_release(payload: object, *, repository: str = REPOSITORY) -> ReleaseInfo:
    """Validate a GitHub release payload and convert it to a ReleaseInfo.

    Parameters
    ----------
    payload : object
        Decoded JSON payload from the GitHub Releases API.
    repository : str
        Repository in ``owner/name`` form.

    Returns
    -------
    ReleaseInfo
        Validated release information.

    Raises
    ------
    ReleaseMetadataError
        If a required field is missing or a URL is not trusted.
    VersionError
        If the tag cannot be parsed as a version.
    """
    if not isinstance(payload, dict):
        raise ReleaseMetadataError("GitHub release metadata is not a JSON object.")

    tag = _require_str(payload, "tag_name")
    if not TAG_PATTERN.match(tag):
        raise ReleaseMetadataError(f"Release tag {tag!r} contains invalid characters.")
    if payload.get("draft") is True:
        raise ReleaseMetadataError("GitHub returned a draft release.")
    version = parse_version_tag(tag)
    if payload.get("prerelease") is True or version.is_prerelease:
        raise ReleaseMetadataError(
            f"The latest GitHub release ({tag}) is a prerelease; "
            "only stable releases are used for updates."
        )

    assets = tuple(_parse_asset(item) for item in _require_list(payload, "assets"))
    return ReleaseInfo(
        tag=tag,
        version=version,
        name=_optional_str(payload, "name") or tag,
        notes=_optional_str(payload, "body") or "",
        page_url=_release_page_url(payload, repository, tag),
        published_at=_optional_str(payload, "published_at"),
        prerelease=False,
        assets=assets,
    )


def parse_version_tag(value: str) -> Version:
    """Parse a release tag or version string, tolerating a leading ``v``.

    Parameters
    ----------
    value : str
        Version string such as ``3.0``, ``v3.0`` or ``v3.1-rc1``.

    Returns
    -------
    Version
        Parsed PEP 440 version.

    Raises
    ------
    VersionError
        If the string is empty or not a valid version.
    """
    if not isinstance(value, str):
        raise VersionError(f"Invalid version value: {value!r}")
    cleaned = value.strip()
    if cleaned[:1] in {"v", "V"}:
        cleaned = cleaned[1:]
    if not cleaned:
        raise VersionError("Version string is empty.")
    try:
        return Version(cleaned)
    except InvalidVersion as error:
        raise VersionError(f"Invalid version string: {value!r}") from error


def _parse_asset(item: object) -> ReleaseAsset:
    """Validate one asset entry."""
    if not isinstance(item, dict):
        raise ReleaseMetadataError("Release asset metadata is not a JSON object.")
    name = _require_str(item, "name")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ReleaseMetadataError(f"Release asset name {name!r} is unsafe.")
    url = _require_str(item, "browser_download_url")
    validate_download_url(url)

    size = item.get("size", 0)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ReleaseMetadataError(f"Release asset {name!r} has an invalid size.")

    digest = item.get("digest")
    sha256: str | None = None
    if isinstance(digest, str):
        match = _DIGEST_PATTERN.match(digest)
        if match is None:
            raise ReleaseMetadataError(
                f"Release asset {name!r} has an unsupported digest format."
            )
        sha256 = match.group(1).lower()
    elif digest is not None:
        raise ReleaseMetadataError(f"Release asset {name!r} has an invalid digest.")

    return ReleaseAsset(name=name, url=url, size=size, sha256=sha256)


def validate_download_url(url: str) -> None:
    """Reject download URLs that are not HTTPS or not on a trusted host.

    Parameters
    ----------
    url : str
        Download URL taken from release metadata.

    Raises
    ------
    ReleaseMetadataError
        If the URL is not a trusted HTTPS URL.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ReleaseMetadataError(f"Update URL is not HTTPS: {url!r}")
    if parsed.hostname not in TRUSTED_HOSTS:
        raise ReleaseMetadataError(f"Update URL host is not trusted: {url!r}")
    if not parsed.path:
        raise ReleaseMetadataError(f"Update URL has no path: {url!r}")


def _release_page_url(payload: Mapping[str, object], repository: str, tag: str) -> str:
    """Return a validated release page URL, falling back to a canonical one."""
    canonical = f"https://github.com/{repository}/releases/tag/{quote(tag)}"
    candidate = payload.get("html_url")
    if not isinstance(candidate, str):
        return canonical
    parsed = urlsplit(candidate)
    expected_prefix = f"/{repository}/releases/".lower()
    if (
        parsed.scheme == "https"
        and parsed.hostname == "github.com"
        and parsed.path.lower().startswith(expected_prefix)
    ):
        return candidate
    return canonical


def _require_str(payload: Mapping[str, object], key: str) -> str:
    """Return a required non-empty string field."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseMetadataError(f"Release metadata is missing a valid {key!r}.")
    return value.strip()


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    """Return an optional string field, or None when absent."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _require_list(payload: Mapping[str, object], key: str) -> list[object]:
    """Return a required list field."""
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ReleaseMetadataError(f"Release metadata field {key!r} is not a list.")
    return value
