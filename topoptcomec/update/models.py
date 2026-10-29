# topoptcomec/update/models.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Data structures shared by the CLI, the GUI, and the update service.

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from packaging.version import Version


@dataclass(frozen=True)
class ReleaseAsset:
    """A downloadable file attached to a GitHub release."""

    name: str
    url: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class ReleaseInfo:
    """Structured view of a GitHub release."""

    tag: str
    version: Version
    name: str
    notes: str
    page_url: str
    published_at: str | None = None
    prerelease: bool = False
    assets: tuple[ReleaseAsset, ...] = field(default_factory=tuple)


class UpdateStatus(enum.Enum):
    """Relationship between the installed and latest stable versions."""

    UP_TO_DATE = "up-to-date"
    UPDATE_AVAILABLE = "update-available"
    INSTALLED_AHEAD = "installed-ahead"


@dataclass(frozen=True)
class UpdateCheck:
    """Result of an update check."""

    installed_version: Version
    latest_version: Version
    status: UpdateStatus
    release: ReleaseInfo

    @property
    def update_available(self) -> bool:
        """Whether a newer stable release can be installed."""
        return self.status is UpdateStatus.UPDATE_AVAILABLE


class InstallationKind(enum.Enum):
    """How the running TopoptComec was installed."""

    GIT_CHECKOUT = "git-checkout"
    FROZEN = "frozen"
    PACKAGE = "package"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Installation:
    """Description of the running installation."""

    kind: InstallationKind
    location: Path | None = None
    detail: str = ""


@dataclass(frozen=True)
class UpdatePlan:
    """How (or whether) an available update can be installed."""

    installation: Installation
    supported: bool
    summary: str
    instructions: str


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of an installation attempt."""

    success: bool
    restart_required: bool
    message: str
