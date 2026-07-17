# topoptcomec/core/grid.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Structured grid: the single source of truth for element/node ordering.

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class StructuredGrid:
    """
    Regular structured grid of unit square (Q4) or cube (H8) elements.

    Flat element ordering (z-major, then x, then y)::

        idx = ez * (nelx * nely) + ex * nely + ey

    Flat node ordering::

        idx = nz * (nelx + 1) * (nely + 1) + nx * (nely + 1) + ny

    Every reshape between flat element vectors and spatial arrays MUST go
    through :meth:`to_spatial` / :meth:`from_spatial` so the ordering is
    defined in exactly one place.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements along each axis. ``nelz == 0`` selects a 2D grid.
    """

    nelx: int
    nely: int
    nelz: int = 0

    def __post_init__(self) -> None:
        if self.nelx <= 0 or self.nely <= 0 or self.nelz < 0:
            raise ValueError(
                f"Invalid grid dimensions: ({self.nelx}, {self.nely}, {self.nelz})"
            )

    # --- Basic properties ---

    @property
    def is_3d(self) -> bool:
        return self.nelz > 0

    @property
    def ndim(self) -> int:
        return 3 if self.is_3d else 2

    @property
    def nel(self) -> int:
        return self.nelx * self.nely * (self.nelz if self.is_3d else 1)

    @property
    def nnodes(self) -> int:
        return (
            (self.nelx + 1) * (self.nely + 1) * ((self.nelz + 1) if self.is_3d else 1)
        )

    @property
    def dofs_per_node(self) -> int:
        return self.ndim

    @property
    def ndof(self) -> int:
        return self.dofs_per_node * self.nnodes

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of a spatial (element) array matching the flat ordering."""
        if self.is_3d:
            return (self.nelz, self.nelx, self.nely)
        return (self.nelx, self.nely)

    # --- Index conversions ---

    def element_index(self, ex: int, ey: int, ez: int = 0) -> int:
        """Flat element index from element coordinates."""
        return (ez * self.nelx * self.nely if self.is_3d else 0) + ex * self.nely + ey

    def node_index(self, nx: int, ny: int, nz: int = 0) -> int:
        """Flat node index from node coordinates."""
        return (
            (nz * (self.nelx + 1) * (self.nely + 1) if self.is_3d else 0)
            + nx * (self.nely + 1)
            + ny
        )

    def element_coordinates(self) -> tuple[IntArray, IntArray, IntArray]:
        """
        Element coordinates (ex, ey, ez) for every flat element index.

        Returns
        -------
        tuple[IntArray, IntArray, IntArray]
            Arrays of shape (nel,); ez is all zeros in 2D.
        """
        el = np.arange(self.nel, dtype=np.int64)
        if self.is_3d:
            ez = el // (self.nelx * self.nely)
            rem = el % (self.nelx * self.nely)
            ex = rem // self.nely
            ey = rem % self.nely
        else:
            ez = np.zeros(self.nel, dtype=np.int64)
            ex = el // self.nely
            ey = el % self.nely
        return ex, ey, ez

    def element_centers(self) -> FloatArray:
        """Element center coordinates, shape (nel, ndim), columns (x, y[, z])."""
        ex, ey, ez = self.element_coordinates()
        cols = [ex + 0.5, ey + 0.5] + ([ez + 0.5] if self.is_3d else [])
        return np.column_stack(cols).astype(np.float64)

    # --- Reshapes ---

    def to_spatial(self, flat: FloatArray) -> FloatArray:
        """Reshape a flat element vector (nel,) to the spatial shape."""
        return np.asarray(flat).reshape(self.spatial_shape)

    def from_spatial(self, spatial: FloatArray) -> FloatArray:
        """Flatten a spatial element array back to flat ordering (nel,)."""
        return np.asarray(spatial).reshape(-1)

    def to_vtk_cell_order(self, flat: npt.NDArray) -> npt.NDArray:
        """Reorder element data so VTK's x-fastest cell indexing is preserved.

        ``flat`` may contain scalar values shaped ``(nel,)`` or values with
        trailing components such as RGB colors shaped ``(nel, 3)``.
        """
        array = np.asarray(flat)
        if array.shape[0] != self.nel:
            raise ValueError(f"Expected {self.nel} elements, got {array.shape[0]}")

        trailing_shape = array.shape[1:]
        spatial = array.reshape((*self.spatial_shape, *trailing_shape))
        trailing_axes = tuple(range(self.ndim, spatial.ndim))
        spatial_axes = (0, 2, 1) if self.is_3d else (1, 0)
        return spatial.transpose((*spatial_axes, *trailing_axes)).reshape(
            (self.nel, *trailing_shape)
        )
