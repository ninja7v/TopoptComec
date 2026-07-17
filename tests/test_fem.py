# tests/test_fem.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the FEM class.

import numpy as np
import pytest
from scipy.sparse import isspmatrix

from topoptcomec.core.fem import FEM
from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Region, Support


# --- Fixtures ---


@pytest.fixture
def fem_2d():
    """Basic 2D FEM: 2x1 mesh with a sensitivity filter."""
    return FEM(
        StructuredGrid(2, 1, 0),
        E=[1.0],
        nu=[0.3],
        penal=3.0,
        solver="Auto",
        filter_type="Sensitivity",
        filter_radius=1.5,
    )


@pytest.fixture
def fem_3d():
    """Basic 3D FEM: 1x1x1 cube."""
    return FEM(
        StructuredGrid(1, 1, 1),
        E=[1.0],
        nu=[0.3],
        penal=3.0,
        filter_radius=1.5,
    )


# --- Tests ---


def test_initialization_2d(fem_2d: FEM):
    """Test if 2D FEM initializes dimensions and degrees of freedom correctly."""
    assert not fem_2d.is_3d
    assert fem_2d.nel == 2 * 1
    assert fem_2d.elemndof == 2
    assert fem_2d.ndof == 2 * (2 + 1) * (1 + 1)
    assert fem_2d.KE.shape == (8, 8)


def test_initialization_3d(fem_3d: FEM):
    """Test if 3D FEM initializes dimensions correctly."""
    assert fem_3d.is_3d
    assert fem_3d.nel == 1
    assert fem_3d.elemndof == 3
    assert fem_3d.ndof == 24
    assert fem_3d.KE.shape == (24, 24)


def test_invalid_solver_rejected():
    """Unknown solver strings must raise instead of silently changing behavior."""
    with pytest.raises(ValueError, match="solver"):
        FEM(StructuredGrid(2, 2, 0), solver="default")


def test_invalid_filter_rejected():
    """Unknown filter type strings must raise."""
    with pytest.raises(ValueError, match="filter_type"):
        FEM(StructuredGrid(2, 2, 0), filter_type=0)


@pytest.mark.parametrize("dims", [(2, 1, 0), (1, 1, 1)])
def test_lk_properties(dims):
    """Element stiffness matrix physical properties (symmetry & equilibrium)."""
    fem = FEM(StructuredGrid(*dims))
    KE = fem.KE

    # 1. Symmetry: K_ij = K_ji
    assert np.allclose(KE, KE.T, atol=1e-10), "The stiffness matrix must be symmetric"

    # 2. Equilibrium: rigid body translation must produce zero force
    assert np.allclose(np.sum(KE, axis=1), 0, atol=1e-10), (
        "The stiffness matrix should satisfy equilibrium (sum of rows = 0)"
    )


def test_boundary_conditions_parsing(fem_2d: FEM):
    """Test if Supports and Loads are parsed into correct DOF indices."""
    # Fix node (0,0) in X and Y; apply force at (2,1) in -Y direction.
    supports = [Support(x=0, y=0, fix_x=True, fix_y=True)]
    loads_in = [Load(x=2, y=1, axis=1, sign=-1, magnitude=1.0)]

    fem_2d.setup_boundary_conditions(loads_in, [], supports)

    assert len(fem_2d.fixed_dofs) == 2, "Should have 2 fixed DOFs for (0,0) in X and Y"
    assert len(fem_2d.loads_in) == 1, "Should have 1 input load"
    assert fem_2d.forces_i.shape == (fem_2d.ndof, 1)
    node = fem_2d.grid.node_index(2, 1)
    assert fem_2d.in_dofs == [2 * node + 1]
    assert fem_2d.forces_i[2 * node + 1, 0] == -1.0


def test_solver_mechanics(fem_2d: FEM):
    """Test that the solver produces a non-zero displacement in the direction of force."""
    # Fix left edge (x=0): nodes (0,0) and (0,1).
    supports = [
        Support(x=0, y=0, fix_x=True, fix_y=True),
        Support(x=0, y=1, fix_x=True, fix_y=True),
    ]
    # Pull right edge (x=2, y=0) to the right (+X); one output port in -Y.
    loads_in = [Load(x=2, y=0, axis=0, sign=1, magnitude=1.0)]
    loads_out = [Load(x=1, y=1, axis=1, sign=-1, magnitude=1.0)]

    fem_2d.setup_boundary_conditions(loads_in, loads_out, supports)

    # Create a solid material (density = 1.0)
    xPhys = np.ones(fem_2d.nel)
    ui, uo = fem_2d.solve(xPhys)

    # Check dimensions
    assert ui.shape == (fem_2d.ndof,)
    assert uo.shape == (fem_2d.ndof,)

    # The node at force application (x=2, y=0) is node index 4 -> DOF 8 (X)
    assert ui[8] > 0.0, (
        "The node under force should have positive displacement in X direction"
    )


def test_sensitivities_calculation(fem_2d: FEM):
    """Test calculation of objective and sensitivities."""
    supports = [Support(x=0, y=0, fix_x=True, fix_y=True)]
    loads_in = [Load(x=2, y=0, axis=0, sign=1, magnitude=1.0)]
    fem_2d.setup_boundary_conditions(loads_in, [], supports)

    xPhys = np.full(fem_2d.nel, 0.5)
    ui, uo = fem_2d.solve(xPhys)

    (dc, dv) = fem_2d.compute_sensitivities(xPhys, ui, uo)
    assert np.all(dc <= 0), (
        "Sensitivities should be negative for compliance minimization"
    )
    assert dc.shape == (fem_2d.nel,), (
        "Sensitivity array should match number of elements"
    )
    assert dv.shape == (fem_2d.nel,), (
        "Volume sensitivities should match number of elements"
    )

    obj = fem_2d.compute_objective(xPhys, ui, uo)
    assert obj > 0, "Compliance objective should be positive"


def test_rigid_objective_uses_full_density_field(fem_2d: FEM):
    """The compliance must weight every element's density.

    The historic implementation indexed xPhys[0] (the density of element 0)
    instead of the density field, producing a wrong reported objective.
    """
    supports = [
        Support(x=0, y=0, fix_x=True, fix_y=True),
        Support(x=0, y=1, fix_x=True, fix_y=True),
    ]
    loads_in = [Load(x=2, y=0, axis=0, sign=1, magnitude=1.0, spring=0.0)]
    fem_2d.setup_boundary_conditions(loads_in, [], supports)

    xPhys = np.array([0.3, 0.9])
    ui, uo = fem_2d.solve(xPhys)
    obj = fem_2d.compute_objective(xPhys, ui, uo)

    # Without springs, compliance == f^T u exactly.
    expected = float(fem_2d.forces_i[:, 0] @ ui)
    np.testing.assert_allclose(obj, expected, rtol=1e-9)


def test_compliant_objective_is_output_displacement(fem_2d: FEM):
    """Compliant objective = signed output displacement under
    the input load (not the self-compliance of the output port)."""
    supports = [
        Support(x=0, y=0, fix_x=True, fix_y=True),
        Support(x=0, y=1, fix_x=True, fix_y=True),
    ]
    loads_in = [Load(x=2, y=0, axis=0, sign=1, magnitude=1.0)]
    loads_out = [Load(x=2, y=1, axis=1, sign=-1, magnitude=1.0)]
    fem_2d.setup_boundary_conditions(loads_in, loads_out, supports)

    xPhys = np.full(fem_2d.nel, 0.8)
    ui, uo = fem_2d.solve(xPhys)
    obj = fem_2d.compute_objective(xPhys, ui, uo)

    out_dof = fem_2d.out_dofs[0]
    expected = -1 * ui[out_dof]  # sign of the output load
    np.testing.assert_allclose(obj, expected, rtol=1e-12)


def test_e_min_scales_with_e_max():
    """E_min must be relative to E_max for conditioning."""
    fem = FEM(StructuredGrid(2, 2, 0), E=[200e9], nu=[0.3])
    np.testing.assert_allclose(fem.E_min, 200e9 * 1e-9)


def test_regions_void(fem_2d: FEM):
    """Test that applying a Void region forces density to near-zero."""
    regions = [Region(shape="box", x=0, y=0, radius=1.0, material=-1)]

    x = np.ones(fem_2d.nel)  # Start fully solid
    x_new = fem_2d.apply_regions(x, regions)

    assert x_new[0] < 0.01, "Element in Void region should have near-zero density"
    assert x_new[1] == 1.0, "Element outside Void region should remain unchanged"


def test_filter_construction(fem_2d: FEM):
    """Test that the filter matrix H is constructed properly."""
    # Check types
    assert isspmatrix(fem_2d.H)
    assert fem_2d.H.shape == (fem_2d.nel, fem_2d.nel)

    # Filter radius is 1.5: elements 0 and 1 are neighbors (dist 1 < 1.5).
    row0 = fem_2d.H.getrow(0).toarray().flatten()
    assert row0[0] > 0, "Element 0 should have a self-connection"
    assert row0[1] > 0, "Element 0 should be connected to neighbor element 1"


def test_boundary_conditions_radius(fem_2d: FEM):
    """Test if a support with radius fixes multiple nodes."""
    # Center at node (1, 0), radius 1: fixes (0,0), (1,0), (2,0), (1,1).
    supports = [Support(x=1, y=0, radius=1.0, fix_x=True, fix_y=True)]
    fem_2d.setup_boundary_conditions([], [], supports)

    fixed_nodes = {int(dof) // fem_2d.dim_mul for dof in fem_2d.fixed_dofs}
    # Node indices: (0,0)->0, (1,0)->2, (1,1)->3, (2,0)->4
    expected_nodes = {0, 2, 3, 4}
    assert fixed_nodes == expected_nodes, (
        f"Expected fixed nodes {expected_nodes}, got {fixed_nodes}"
    )


def test_structured_grid_vtk_cell_order():
    """VTK conversion preserves element coordinates for scalar and RGB data."""
    grid = StructuredGrid(2, 3, 2)
    values = np.arange(grid.nel)
    expected = np.array(
        [
            grid.element_index(ex, ey, ez)
            for ez in range(grid.nelz)
            for ey in range(grid.nely)
            for ex in range(grid.nelx)
        ]
    )

    np.testing.assert_array_equal(grid.to_vtk_cell_order(values), expected)
    rgb = np.column_stack((values, values + grid.nel, values + 2 * grid.nel))
    np.testing.assert_array_equal(grid.to_vtk_cell_order(rgb), rgb[expected])
