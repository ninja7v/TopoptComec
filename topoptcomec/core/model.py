# topoptcomec/core/model.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Typed problem definition: loads, supports and geometric regions.

from __future__ import annotations
from dataclasses import dataclass

#: Axis identifiers used by :class:`Load`.
AXIS_X, AXIS_Y, AXIS_Z = 0, 1, 2


@dataclass(frozen=True)
class Load:
    """
    Point load applied at a grid node.

    Notes
    -----
    Following the classic compliant-mechanism formulation (Sigmund 1997),
    an artificial spring of stiffness ``spring`` is attached to the loaded
    DOF. When ``spring`` is None, the force magnitude is reused as the
    spring stiffness (the historic TopoptComec convention).

    Axis convention: the grid is unit-element based; ``sign=+1`` points
    toward increasing coordinates (x right, y down on screen, z away).
    """

    x: int
    y: int
    z: int = 0
    axis: int = AXIS_X  # 0=x, 1=y, 2=z
    sign: int = 1  # +1 along axis, -1 opposite
    magnitude: float = 1.0
    spring: float | None = None

    @property
    def spring_stiffness(self) -> float:
        return self.magnitude if self.spring is None else self.spring

    def signed_magnitude(self) -> float:
        return self.sign * self.magnitude


@dataclass(frozen=True)
class Support:
    """Fixed node (optionally all nodes within ``radius``) per axis."""

    x: int
    y: int
    z: int = 0
    radius: float = 0.0
    fix_x: bool = False
    fix_y: bool = False
    fix_z: bool = False


@dataclass(frozen=True)
class Region:
    """
    Geometric density constraint: force a sphere/box to be void or solid.

    ``shape`` is either ``"sphere"`` or ``"box"``; ``radius`` is the sphere
    radius or the box half-width.
    """

    shape: str  # "sphere" | "box"
    x: float
    y: float
    z: float = 0.0
    radius: float = 0.0
    solid: bool = True

    def __post_init__(self) -> None:
        if self.shape not in ("sphere", "box"):
            raise ValueError(f"Unknown region shape: {self.shape!r}")
