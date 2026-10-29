# topoptcomec/update/transport.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Minimal HTTP transport used by the update service.
#
# Only the standard library is used so the service works without PySide6 and
# without adding an HTTP client dependency.

from __future__ import annotations

import http.client
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import BinaryIO, Protocol

from .errors import NetworkError, RateLimitError, ReleaseMetadataError

#: Artificial ceiling for release metadata, which is a few hundred KiB at most.
MAX_JSON_BYTES = 4 * 1024 * 1024


class Transport(Protocol):
    """HTTP operations required by the update service.

    Tests inject a fake implementation so no test touches the network.
    """

    def get_json(
        self, url: str, timeout: float, headers: Mapping[str, str] | None = None
    ) -> tuple[int, Mapping[str, str], bytes]:
        """Fetch a URL and return status, headers, and body."""

    def open_stream(self, url: str, timeout: float) -> BinaryIO:
        """Open a URL for streaming download."""


class UrllibTransport:
    """Transport backed by :mod:`urllib` with TLS verification left enabled."""

    def get_json(
        self, url: str, timeout: float, headers: Mapping[str, str] | None = None
    ) -> tuple[int, Mapping[str, str], bytes]:
        """Fetch a URL and return status, headers, and body."""
        request = urllib.request.Request(url, headers=dict(headers or {}))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body: bytes = response.read(MAX_JSON_BYTES)
                return int(response.status), dict(response.headers), body
        except urllib.error.HTTPError as error:
            typed = _http_error(error)
            error.close()
            raise typed from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise NetworkError(f"Could not reach {url}: {error}") from error
        except http.client.HTTPException as error:
            raise NetworkError(f"Invalid HTTP response from {url}: {error}") from error

    def open_stream(self, url: str, timeout: float) -> BinaryIO:
        """Open a URL for streaming download."""
        request = urllib.request.Request(
            url, headers={"User-Agent": "TopoptComec-updater"}
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            typed = _http_error(error)
            error.close()
            raise typed from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise NetworkError(f"Could not download {url}: {error}") from error


def _http_error(error: urllib.error.HTTPError) -> NetworkError:
    """Translate an HTTP error into a typed update error."""
    remaining = (
        str(error.headers.get("X-RateLimit-Remaining", "")) if error.headers else ""
    )
    if error.code == 403 and remaining == "0":
        return RateLimitError(
            "GitHub API rate limit exceeded. Try again later or download the "
            "release manually from the project page."
        )
    if error.code == 404:
        return ReleaseMetadataError("No stable release was found on GitHub (HTTP 404).")
    return NetworkError(f"GitHub request failed with HTTP {error.code}.")
