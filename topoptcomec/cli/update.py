# topoptcomec/cli/update.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# ``topoptcomec update`` command.

from __future__ import annotations

import argparse
import sys

from topoptcomec.update import (
    GithubReleaseClient,
    InstallationError,
    RateLimitError,
    UpdateCheck,
    UpdateError,
    UpdateStatus,
    build_plan,
    check_for_update,
    detect_installation,
    install_update,
)
from topoptcomec.version import __version__

#: The update check and installation completed, or nothing needed to change.
EXIT_OK = 0
#: A network, metadata, or installation error occurred.
EXIT_ERROR = 1
#: Confirmation was required but the session is not interactive.
EXIT_CONFIRMATION_REQUIRED = 2
#: An update exists but must be installed manually.
EXIT_MANUAL_ACTION = 3

_DEFAULT_TIMEOUT = 15.0


def _build_update_parser() -> argparse.ArgumentParser:
    """Build the parser for ``topoptcomec update``."""
    parser = argparse.ArgumentParser(
        prog="topoptcomec update",
        description=(
            "Check GitHub for the latest stable TopoptComec release and install "
            "it when the current installation supports automatic updates."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether an update is available; do not install it",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Install without asking for confirmation (non-interactive use)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help=f"Network timeout in seconds (default: {_DEFAULT_TIMEOUT:g})",
    )
    return parser


def run_update_command(argv: list[str] | None = None) -> int:
    """Run the update command and return its exit code.

    Parameters
    ----------
    argv : list[str] | None
        Arguments after ``topoptcomec update``. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success or when no update is needed, ``1`` on error, ``2``
        when confirmation is needed in a non-interactive session, ``3`` when
        the update must be installed manually.
    """
    args = _build_update_parser().parse_args(argv)
    if args.timeout <= 0:
        print("Error: --timeout must be greater than zero.", file=sys.stderr)
        return EXIT_ERROR

    print(f"Installed version: {__version__}")
    print("Checking GitHub for the latest stable release...")
    try:
        check = check_for_update(client=GithubReleaseClient(timeout=args.timeout))
    except RateLimitError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except UpdateError as error:
        print(f"Error: could not check for updates: {error}", file=sys.stderr)
        return EXIT_ERROR

    _print_versions(check)
    if not check.update_available:
        return EXIT_OK

    _print_release(check)
    installation = detect_installation()
    plan = build_plan(check, installation)
    print(f"Installation method: {installation.detail}")
    if installation.location is not None:
        print(f"Location: {installation.location}")
    print(f"Update method: {plan.summary}")

    if args.check:
        print(f"Update available: {check.latest_version} (not installed).")
        return EXIT_OK
    if not plan.supported:
        print(
            "Automatic installation is not supported for this installation.",
            file=sys.stderr,
        )
        print(plan.instructions)
        return EXIT_MANUAL_ACTION
    if not args.yes and not _stdin_is_interactive():
        print(
            "Error: confirmation is required but stdin is not interactive. "
            "Re-run with --yes to install non-interactively.",
            file=sys.stderr,
        )
        return EXIT_CONFIRMATION_REQUIRED
    if not args.yes:
        reply = input("Install this update now? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Update cancelled. Nothing was changed.")
            return EXIT_OK

    try:
        result = install_update(check, plan, progress=_print_progress)
    except InstallationError as error:
        print(f"Error: update failed: {error}", file=sys.stderr)
        print(
            "The existing installation was left in place. Retry later, or "
            f"install manually from {check.release.page_url}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    except UpdateError as error:
        print(f"Error: update failed: {error}", file=sys.stderr)
        print("The existing installation was left in place.", file=sys.stderr)
        return EXIT_ERROR

    print(result.message)
    if result.restart_required:
        print("Restart TopoptComec to use the new version.")
    return EXIT_OK


def _print_versions(check: UpdateCheck) -> None:
    """Print the installed and latest versions and the comparison result."""
    print(f"Latest stable version: {check.latest_version} ({check.release.tag})")
    if check.status is UpdateStatus.INSTALLED_AHEAD:
        print(
            f"Installed version {check.installed_version} is newer than the "
            "latest release; no update needed."
        )
    elif check.status is UpdateStatus.UP_TO_DATE:
        print(f"TopoptComec {check.installed_version} is up to date.")
    else:
        print(
            f"Update available: {check.latest_version} "
            f"(installed: {check.installed_version})"
        )


def _print_release(check: UpdateCheck) -> None:
    """Print release name, page, and notes."""
    print(f"Release: {check.release.name}")
    print(f"Release page: {check.release.page_url}")
    print("Release notes:")
    notes = check.release.notes.strip()
    if not notes:
        print("  (no release notes provided)")
        return
    for line in notes.splitlines():
        print(f"  {line}" if line else "")


def _print_progress(percent: int, message: str) -> None:
    """Print installation progress when the stage changes."""
    print(f"  [{percent:3d}%] {message}")


def _stdin_is_interactive() -> bool:
    """Whether standard input can be used for a confirmation prompt."""
    stream = sys.stdin
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())
