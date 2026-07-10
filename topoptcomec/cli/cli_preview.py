# topoptcomec/cli_preview.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Terminal preview renderer for CLI results.

from __future__ import annotations

import shutil
from matplotlib.colors import to_rgb

import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float64]

# Unicode half-block characters allow encoding two vertical pixels per
# character cell: the upper-half block sets the foreground (top pixel)
# while the background colour represents the bottom pixel.
_UPPER_HALF = "\u2580"  # ▀

_RESET = "\033[0m"


def _ansi_rgb_fg(r: int, g: int, b: int) -> str:
    """Return an ANSI escape that sets a 24-bit foreground colour."""
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"\033[38;2;{r};{g};{b}m"


def _ansi_rgb_bg(r: int, g: int, b: int) -> str:
    """Return an ANSI escape that sets a 24-bit background colour."""
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"\033[48;2;{r};{g};{b}m"


def _resolve_colors(colors: list[str] | None, n_mat: int) -> np.ndarray:
    """
    Return an ``(n_mat, 3)`` RGB array (float [0, 1]) from hex color strings.

    If *colors* is ``None`` or empty, defaults to black for every
    material.
    """
    if colors is None or len(colors) == 0:
        colors = ["#000000"] * n_mat
    while len(colors) < n_mat:
        colors.append("#000000")
    rgb = np.array([to_rgb(c) for c in colors[:n_mat]], dtype=np.float64)
    return rgb


def _density_to_2d_rgb(
    xPhys: FloatArray,
    nelxyz: list[int],
    colors: list[str] | None,
) -> FloatArray:
    """
    Convert a density field into a 2-D RGB image array (rows × cols × 3).

    For multi-material fields each material slice contributes its own
    colour, blended additively (clipped to [0, 1]).  Void elements
    (total density ≈ 0) appear white, matching the GUI plot background.

    For 3-D problems the XY plane is obtained via a max-projection
    along the Z axis (applied per-material so colours are preserved).

    The returned array has shape ``(ny, nx, 3)`` with the Y axis running
    top to bottom (row 0 = maximum y, matching how a terminal prints).

    Parameters
    ----------
    xPhys : FloatArray
        Flat element densities, shape ``(nel,)`` or ``(n_mat, nel)``.
    nelxyz : list[int]
        ``[nx, ny, nz]`` element counts.
    colors : list[str] | None
        Hex colour strings, one per material.

    Returns
    -------
    FloatArray
        3-D array with shape ``(ny, nx, 3)``, values in [0, 1].
    """
    nx, ny, nz = nelxyz
    is_multi = xPhys.ndim == 2
    n_mat = xPhys.shape[0] if is_multi else 1
    mat_rgb = _resolve_colors(colors, n_mat)

    if is_multi:
        planes = np.zeros((n_mat, nx, ny), dtype=np.float64)
        for i in range(n_mat):
            field_i = xPhys[i]
            if nz > 0:
                vol = field_i.reshape((nz, nx, ny))
                planes[i] = vol.max(axis=0)
            else:
                planes[i] = field_i.reshape((nx, ny))

        total_rho = np.clip(planes.sum(axis=0), 0.0, 1.0)  # (nx, ny)
        rgb = np.ones((nx, ny, 3), dtype=np.float64)  # white background
        rgb *= (1.0 - total_rho)[:, :, np.newaxis]
        for i in range(n_mat):
            rgb += planes[i][:, :, np.newaxis] * mat_rgb[i][np.newaxis, np.newaxis, :]
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        field = xPhys
        if nz > 0:
            vol = field.reshape((nz, nx, ny))
            plane = vol.max(axis=0)
        else:
            plane = field.reshape((nx, ny))
        rgb = np.ones((nx, ny, 3), dtype=np.float64)
        rgb = rgb * (1.0 - plane)[:, :, np.newaxis]
        rgb += plane[:, :, np.newaxis] * mat_rgb[0][np.newaxis, np.newaxis, :]

    image = np.transpose(rgb, (1, 0, 2))  # (ny, nx, 3)
    image = np.flipud(image)
    return np.clip(image, 0.0, 1.0)


def _downscale_rgb(image: FloatArray, target_cols: int) -> FloatArray:
    """
    Downscale an RGB image so its width equals *target_cols*.

    Uses bilinear interpolation along columns then rows.  The height is
    scaled by the same factor to preserve the aspect ratio.

    Parameters
    ----------
    image : FloatArray
        Source image with shape ``(rows, cols, 3)``.
    target_cols : int
        Desired number of columns in the output.

    Returns
    -------
    FloatArray
        Downscaled image with shape ``(target_rows, target_cols, 3)``.
    """
    rows, cols, _ = image.shape
    if cols <= target_cols:
        return image

    scale = target_cols / cols
    target_rows = max(1, int(rows * scale))
    target_cols = max(1, target_cols)

    xs_new = np.linspace(0, cols - 1, target_cols)
    ys_new = np.linspace(0, rows - 1, target_rows)

    col_indices = np.clip(xs_new.astype(int), 0, cols - 2)
    col_frac = xs_new - col_indices
    interp_cols = (
        image[:, col_indices, :] * (1 - col_frac[np.newaxis, :, np.newaxis])
        + image[:, col_indices + 1, :] * col_frac[np.newaxis, :, np.newaxis]
    )

    row_indices = np.clip(ys_new.astype(int), 0, rows - 2)
    row_frac = ys_new - row_indices
    result = (
        interp_cols[row_indices, :, :] * (1 - row_frac[:, np.newaxis, np.newaxis])
        + interp_cols[row_indices + 1, :, :] * row_frac[:, np.newaxis, np.newaxis]
    )
    return np.clip(result, 0.0, 1.0)


def _render_lines_half_block(image: FloatArray) -> list[str]:
    """
    Render an RGB *image* as terminal lines using half-block characters.

    Each output line encodes **two** pixel rows using the Unicode
    upper-half-block character (▀) with 24-bit ANSI colours: the
    foreground colour represents the top pixel and the background
    colour represents the bottom pixel.

    When the image has an odd number of rows the last row is paired
    with a black (empty) row.

    Parameters
    ----------
    image : FloatArray
        3-D array ``(rows, cols, 3)`` with values in [0, 1].

    Returns
    -------
    list[str]
        One string per output line (no trailing newline).
    """
    rows, cols, _ = image.shape

    if rows % 2 != 0:
        image = np.vstack([image, np.zeros((1, cols, 3))])
        rows += 1

    lines: list[str] = []
    for r in range(0, rows, 2):
        top_row = image[r]
        bot_row = image[r + 1]
        parts: list[str] = []
        for c in range(cols):
            tr, tg, tb = (int(top_row[c, ch] * 255) for ch in range(3))
            br, bg, bb = (int(bot_row[c, ch] * 255) for ch in range(3))
            parts.append(
                f"{_ansi_rgb_fg(tr, tg, tb)}{_ansi_rgb_bg(br, bg, bb)}{_UPPER_HALF}"
            )
        parts.append(_RESET)
        lines.append("".join(parts))
    return lines


def render_preview(
    xPhys: FloatArray,
    nelxyz: list[int],
    *,
    colors: list[str] | None = None,
    max_width: int | None = None,
) -> str:
    """
    Build a complete terminal preview string for a density field.

    The output is strictly line-based: each row is printed on its own
    line with a fixed width so that terminal resizing does not distort
    the layout.

    Each pixel is rendered with the colour of its material on a white
    background, matching the GUI matplotlib plot.  When *colors* is
    ``None`` or empty, all materials default to black.

    Parameters
    ----------
    xPhys : FloatArray
        Flat element densities (1-D or 2-D for multi-material).
    nelxyz : list[int]
        ``[nx, ny, nz]`` element counts.
    colors : list[str] | None
        Hex colour strings for each material.  If ``None``, materials
        default to black.
    max_width : int | None
        Maximum number of columns to use.  Defaults to the current
        terminal width.

    Returns
    -------
    str
        Multi-line string ready to be printed.
    """
    if max_width is None:
        max_width = shutil.get_terminal_size((80, 24)).columns

    max_width = max(max_width, 10)

    is_3d = nelxyz[2] > 0
    image = _density_to_2d_rgb(xPhys, nelxyz, colors)
    image = _downscale_rgb(image, max_width)
    lines = _render_lines_half_block(image)

    nx, ny, nz = nelxyz
    display_h, display_w = image.shape[0], image.shape[1]
    if is_3d:
        header = (
            f"Preview ({nx}×{ny}×{nz} → XY projection, {display_w}×{display_h} display)"
        )
    else:
        header = f"Preview ({nx}×{ny} → {display_w}×{display_h} display)"

    if len(header) > max_width:
        header = header[: max_width - 1] + "…"

    separator = "─" * min(len(header), max_width)

    return "\n".join([separator, header, separator, *lines, separator])
