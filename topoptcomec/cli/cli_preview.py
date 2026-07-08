# topoptcomec/cli_preview.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Terminal preview renderer for CLI results.

from __future__ import annotations

import os
import shutil

import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float64]

# Unicode half-block characters allow encoding two vertical pixels per
# character cell: the upper-half block sets the foreground (top pixel)
# while the background colour represents the bottom pixel.
_UPPER_HALF = "\u2580"  # ▀
_LOWER_HALF = "\u2584"  # ▄
_FULL_BLOCK = "\u2588"  # █

# Quantised grey ramp used for the density values (8 levels).
_GREY_RAMP = " ░▒▓█"


def _ansi_grey_fg(level: int) -> str:
    """Return an ANSI escape that sets a 24-bit foreground grey."""
    level = max(0, min(255, level))
    return f"\033[38;2;{level};{level};{level}m"


def _ansi_grey_bg(level: int) -> str:
    """Return an ANSI escape that sets a 24-bit background grey."""
    level = max(0, min(255, level))
    return f"\033[48;2;{level};{level};{level}m"


_RESET = "\033[0m"


def _density_to_2d(xPhys: FloatArray, nelxyz: list[int]) -> FloatArray:
    """
    Convert a flat density field into a 2-D image array (rows × cols).

    For 2-D problems (nz == 0) the field is reshaped directly.
    For 3-D problems the XY plane is obtained via a max-projection
    along the Z axis.

    The returned array has shape (ny, nx) with the Y axis running top
    to bottom (row 0 = maximum y, matching how a terminal prints).

    Parameters
    ----------
    xPhys : FloatArray
        Flat element densities, shape ``(nel,)`` or ``(n_mat, nel)``.
    nelxyz : list[int]
        ``[nx, ny, nz]`` element counts.

    Returns
    -------
    FloatArray
        2-D array with shape ``(ny, nx)``, values in [0, 1].
    """
    nx, ny, nz = nelxyz
    is_multi = xPhys.ndim == 2

    # Collapse multi-material to effective density
    field: FloatArray = xPhys.sum(axis=0) if is_multi else xPhys

    if nz > 0:
        # 3-D: index convention is  z * (nx * ny) + x * ny + y
        vol = field.reshape((nz, nx, ny))  # shape (nz, nx, ny)
        plane = vol.max(axis=0)  # max-project along Z → (nx, ny)
    else:
        plane = field.reshape((nx, ny))  # (nx, ny)

    # Transpose so that rows = y, cols = x and flip y so that the
    # highest y is at the top (matches origin="lower" convention).
    image: FloatArray = np.flipud(plane.T)  # (ny, nx)

    return np.clip(image, 0.0, 1.0)


def _downscale(image: FloatArray, target_cols: int) -> FloatArray:
    """
    Downscale *image* so that its width equals *target_cols*.

    The height is scaled by the same factor to preserve the aspect
    ratio.  Uses area-averaging (block mean) for clean downscaling.

    Parameters
    ----------
    image : FloatArray
        Source image with shape ``(rows, cols)``.
    target_cols : int
        Desired number of columns in the output.

    Returns
    -------
    FloatArray
        Downscaled image.
    """
    rows, cols = image.shape
    if cols <= target_cols:
        return image

    scale = target_cols / cols
    target_rows = max(1, int(rows * scale))
    target_cols = max(1, target_cols)

    # Resize via block-mean: reshape into blocks and average.
    # For non-divisible sizes we use np.interp-based scaling which is
    # lightweight and does not require scipy or PIL.
    xs_new = np.linspace(0, cols - 1, target_cols)
    ys_new = np.linspace(0, rows - 1, target_rows)

    # Bilinear interpolation along columns, then rows.
    col_indices = np.clip(xs_new.astype(int), 0, cols - 2)
    col_frac = xs_new - col_indices
    interp_cols = (
        image[:, col_indices] * (1 - col_frac[np.newaxis, :])
        + image[:, col_indices + 1] * col_frac[np.newaxis, :]
    )

    row_indices = np.clip(ys_new.astype(int), 0, rows - 2)
    row_frac = ys_new - row_indices
    result = (
        interp_cols[row_indices, :] * (1 - row_frac[:, np.newaxis])
        + interp_cols[row_indices + 1, :] * row_frac[:, np.newaxis]
    )
    return np.clip(result, 0.0, 1.0)


def _render_lines_half_block(image: FloatArray) -> list[str]:
    """
    Render *image* as terminal lines using half-block characters.

    Each output line encodes **two** pixel rows using the Unicode
    upper-half-block character (▀) with 24-bit ANSI colours: the
    foreground colour represents the top pixel and the background
    colour represents the bottom pixel.

    When the image has an odd number of rows the last row is paired
    with a black (empty) row.

    Parameters
    ----------
    image : FloatArray
        2-D array (rows, cols) with values in [0, 1].

    Returns
    -------
    list[str]
        One string per output line (no trailing newline).
    """
    rows, cols = image.shape

    # Pad to even row count
    if rows % 2 != 0:
        image = np.vstack([image, np.zeros((1, cols))])
        rows += 1

    lines: list[str] = []
    for r in range(0, rows, 2):
        top_row = image[r]
        bot_row = image[r + 1]
        parts: list[str] = []
        for c in range(cols):
            fg = int(top_row[c] * 255)
            bg = int(bot_row[c] * 255)
            parts.append(f"{_ansi_grey_fg(fg)}{_ansi_grey_bg(bg)}{_UPPER_HALF}")
        parts.append(_RESET)
        lines.append("".join(parts))
    return lines


def _render_lines_ascii(image: FloatArray) -> list[str]:
    """
    Render *image* as terminal lines using ASCII block characters.

    Provides a fallback for terminals that lack true-colour support.
    Each pixel maps to one character from the ``_GREY_RAMP`` string.

    Parameters
    ----------
    image : FloatArray
        2-D array (rows, cols) with values in [0, 1].

    Returns
    -------
    list[str]
        One string per output line (no trailing newline).
    """
    max_idx = len(_GREY_RAMP) - 1
    indices = np.clip((image * max_idx).astype(int), 0, max_idx)
    lines: list[str] = []
    for r in range(indices.shape[0]):
        lines.append("".join(_GREY_RAMP[i] for i in indices[r]))
    return lines


def render_preview(
    xPhys: FloatArray,
    nelxyz: list[int],
    *,
    use_color: bool | None = None,
    max_width: int | None = None,
) -> str:
    """
    Build a complete terminal preview string for a density field.

    The output is strictly line-based: each row is printed on its own
    line with a fixed width so that terminal resizing does not distort
    the layout.

    Parameters
    ----------
    xPhys : FloatArray
        Flat element densities (1-D or 2-D for multi-material).
    nelxyz : list[int]
        ``[nx, ny, nz]`` element counts.
    use_color : bool | None
        If ``True``, use 24-bit ANSI colour half-block rendering.
        If ``False``, use plain ASCII characters.  If ``None``
        (default), auto-detect based on the ``TERM`` / ``NO_COLOR``
        environment variables and ``os.isatty``.
    max_width : int | None
        Maximum number of columns to use.  Defaults to the current
        terminal width.

    Returns
    -------
    str
        Multi-line string ready to be printed.
    """
    # Detect terminal capabilities
    if use_color is None:
        use_color = (
            os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM", "") != "dumb"
            and hasattr(os, "isatty")
            and os.isatty(1)
        )

    if max_width is None:
        max_width = shutil.get_terminal_size((80, 24)).columns

    # Ensure at least a tiny preview
    max_width = max(max_width, 10)

    is_3d = nelxyz[2] > 0
    image = _density_to_2d(xPhys, nelxyz)
    image = _downscale(image, max_width)

    if use_color:
        lines = _render_lines_half_block(image)
    else:
        lines = _render_lines_ascii(image)

    # Build header
    nx, ny, nz = nelxyz
    if is_3d:
        header = f"Preview ({nx}×{ny}×{nz} → XY projection, {image.shape[1]}×{image.shape[0]} display)"
    else:
        header = f"Preview ({nx}×{ny} → {image.shape[1]}×{image.shape[0]} display)"

    # Truncate header to fit within max_width
    if len(header) > max_width:
        header = header[: max_width - 1] + "…"

    separator = "─" * min(len(header), max_width)

    return "\n".join([separator, header, separator, *lines, separator])
