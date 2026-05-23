# tests/test_fem.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the FEM class.

import numpy as np
import pytest
from scipy.sparse import isspmatrix

from app.core.fem import FEM


# --- Fixtures ---


@pytest.fixture
def base_config_2d():
    """Basic 2D configuration: 2x1 Mesh."""
    return {
        "Dimensions": {"nelxyz": [2, 1, 0], "volfrac": 0.5},
        "Materials": {"E": [1.0], "nu": [0.3], "init_type": 0},
        "Optimizer": {
            "penal": 3.0,
            "filter_radius_min": 1.5,
            "solver": "default",
            "filter_type": 0,  # Sensitivity filter
        },
    }


@pytest.fixture
def base_config_3d():
    """Basic 3D configuration: 1x1x1 Cube."""
    return {
        "Dimensions": {"nelxyz": [1, 1, 1], "volfrac": 0.5},
        "Materials": {"E": [1.0], "nu": [0.3]},
        "Optimizer": {"penal": 3.0, "filter_radius_min": 1.5},
    }


# --- Tests ---


def test_initialization_2d(base_config_2d):
    """Test if 2D FEM initializes dimensions and degrees of freedom correctly."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )

    assert not fem.is_3d
    assert fem.nel == 2 * 1
    assert fem.elemndof == 2
    assert fem.ndof == 2 * (2 + 1) * (1 + 1)
    assert fem.KE.shape == (8, 8)


def test_initialization_3d(base_config_3d):
    """Test if 3D FEM initializes dimensions correctly."""
    fem = FEM(
        base_config_3d["Dimensions"],
        base_config_3d["Materials"],
        base_config_3d["Optimizer"],
    )

    assert fem.is_3d
    assert fem.nel == 1
    assert fem.elemndof == 3
    assert fem.ndof == 24
    assert fem.KE.shape == (24, 24)


def test_lk_properties_2d(base_config_2d):
    """Test physical properties of the element stiffness matrix (Symmetry & Equilibrium)."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )
    KE = fem.KE

    # 1. Symmetry: K_ij = K_ji
    assert np.allclose(KE, KE.T, atol=1e-10), "The stiffness matrix must be symmetric"
    assert KE.shape == (8, 8), "The 2D stiffness matrix must be 8x8"

    # 2. Equilibrium: Sum of rows/cols should be zero (rigid body motion results in 0 force)
    # Summing columns implies applying unit displacement to all DOFs -> Force should be 0
    assert np.allclose(np.sum(KE, axis=1), 0, atol=1e-10), (
        "The stiffness matrix should satisfy equilibrium (sum of rows = 0)"
    )


def test_boundary_conditions_parsing(base_config_2d):
    """Test if Supports and Forces are parsed into correct DOF indices."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )

    # Scenario:
    # Fix Node at (0,0) in X and Y.
    # Apply Force at (2,1) in Y direction.
    Supports = {"sx": [0], "sy": [0], "sz": [], "sdim": ["XY"]}
    Forces = {
        "fix": [2],
        "fiy": [1],
        "fiz": [],
        "fidir": ["Y:\u2191"],
        "finorm": [1.0],
        # Output forces (required by parse logic even if empty)
        "fox": [],
        "foy": [],
        "foz": [],
        "fodir": [],
        "fonorm": [],
    }

    fem.setup_boundary_conditions(Forces, Supports)

    assert len(fem.fixed_dofs) == 2, "Should have 2 fixed DOFs for (0,0) in X and Y"
    assert len(fem.fi_indices) == 1, "Should have 1 input force DOF for (2,1) in Y"


def test_solver_mechanics(base_config_2d):
    """Test that the solver produces a non-zero displacement in the direction of force."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )

    # Fix left edge (x=0)
    # Nodes at x=0 are (0,0) and (0,1). Indices: 0*(2)+0=0, 0*(2)+1=1.
    Supports = {"sx": [0, 0], "sy": [0, 1], "sz": [], "sdim": ["XY", "XY"]}
    # Pull right edge (x=2, y=0) to the right (X)
    Forces = {
        "fix": [0, 2],
        "fiy": [0, 0],
        "fiz": [],
        "fidir": ["-", "X:\u2192"],
        "finorm": [1.0, 1.0],
        "fox": [0, 1],
        "foy": [0, 1],
        "foz": [],
        "fodir": ["-", "Y:\u2191"],
        "fonorm": [1.0, 1.0],
    }

    fem.setup_boundary_conditions(Forces, Supports)

    # Create a solid material (density = 1.0)
    xPhys = np.ones(fem.nel)

    # Solve
    ui, uo = fem.solve(xPhys)

    # Check dimensions
    assert ui.shape == (fem.ndof, 1)
    assert uo.shape == (fem.ndof, 1)

    # The node at force application (x=2, y=0) is Node index 4 -> DOF 8 (X)
    assert ui[8, 0] > 0.0, (
        "The node under force should have positive displacement in X direction"
    )


def test_sensitivities_calculation(base_config_2d):
    """Test calculation of objective and sensitivities."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )

    # Minimal BCs to ensure stability
    Supports = {"sx": [0], "sy": [0], "sz": [], "sdim": ["XY"]}
    Forces = {
        "fix": [2, 0],
        "fiy": [0, 0],
        "fiz": [],
        "fidir": ["X:\u2192", "-"],
        "finorm": [1.0, 1.0],
        "fox": [],
        "foy": [],
        "foz": [],
        "fodir": [],
        "fonorm": [],
    }
    fem.setup_boundary_conditions(Forces, Supports)

    xPhys = np.full(fem.nel, 0.5)
    ui, uo = fem.solve(xPhys)

    (dc, dv) = fem.compute_sensitivities(xPhys, ui, uo)
    assert np.all(dc <= 0), (
        "Sensitivities should be negative for compliance minimization"
    )
    assert dc.shape == (fem.nel,), "Sensitivity array should match number of elements"
    assert dv.shape == (fem.nel,), (
        "Volume sensitivities should match number of elements"
    )

    obj = fem.compute_objective(xPhys, ui, uo)
    assert obj > 0, "Compliance objective should be positive"


def test_regions_void(base_config_2d):
    """Test that applying a Void region forces density to near-zero."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )

    # Define a void region at x=0, y=0
    Regions = {
        "rx": [0, 0],
        "ry": [0, 0],
        "rz": [],
        "rradius": [1, 1],  # Small radius covering the element center
        "rshape": ["-", "□"],  # Square
        "rstate": ["Void", "Void"],
    }

    x = np.ones(fem.nel)  # Start fully solid
    x_new = fem.apply_regions(x, Regions)

    assert x_new[0] < 0.01, "Element in Void region should have near-zero density"
    assert x_new[1] == 1.0, "Element outside Void region should remain unchanged"


def test_filter_construction(base_config_2d):
    """Test that the filter matrix H is constructed properly."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )

    # Check types
    assert isspmatrix(fem.H)
    assert fem.H.shape == (fem.nel, fem.nel)

    # Filter radius is 1.5.
    # Element 0 (0,0) should be connected to Element 1 (0,1) because dist=1 < 1.5.
    # It should not be connected to Element at (1,0) if elements are unit size?
    # Note: In the code logic, distance is between element indices in grid.
    # (0,0) and (0,1) are neighbors.

    # Get the row for element 0
    row0 = fem.H.getrow(0).toarray().flatten()
    assert row0[0] > 0, "Element 0 should have a self-connection"
    assert row0[1] > 0, "Element 0 should be connected to neighbor element 1"


def test_boundary_conditions_radius(base_config_2d):
    """Test if a support with radius fixes multiple nodes."""
    fem = FEM(
        base_config_2d["Dimensions"],
        base_config_2d["Materials"],
        base_config_2d["Optimizer"],
    )

    # Center at (1, 0) (bottom middle node)
    # Radius 1 -> should cover (0,0), (1,0), (2,0) and maybe (0,1), (1,1), (2,1)?
    # Dist from (1,0) to (0,0) is 1. <= 1 -> Yes.
    # Dist from (1,0) to (1,1) is 1. <= 1 -> Yes.
    Supports = {
        "sx": [1],
        "sy": [0],
        "sz": [],
        "sdim": ["XY"],
        "sr": [1],
    }
    Forces = {
        "fix": [],
        "fiy": [],
        "fiz": [],
        "fidir": [],
        "finorm": [],
        "fox": [],
        "foy": [],
        "foz": [],
        "fodir": [],
        "fonorm": [],
    }

    fem.setup_boundary_conditions(Forces, Supports)

    # Nodes:
    # (0,0) -> 0
    # (1,0) -> 1 (Center)
    # (2,0) -> 2
    # (0,1) -> 3
    # (1,1) -> 4
    # (2,1) -> 5

    # Check fixed DOFs
    # 0, 1, 2, 4 are within radius 1 of (1,0)
    # 0: dist=1
    # 2: dist=1
    # 4: dist=1 (x=1, y=1) -> sqrt(0^2 + 1^2) = 1
    # 3: dist=sqrt(1^2+1^2)=1.41 > 1 -> No
    # 5: dist=sqrt(1^2+1^2)=1.41 > 1 -> No

    fixed_nodes = set()
    for dof in fem.fixed_dofs:
        fixed_nodes.add(dof // fem.dim_mul)

    expected_nodes = {0, 2, 3, 4}
    assert fixed_nodes == expected_nodes, (
        f"Expected fixed nodes {expected_nodes}, got {fixed_nodes}"
    )
