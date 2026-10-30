# tests/test_exporters.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the exporters.

from pathlib import Path

import numpy as np

from topoptcomec import exporters


def test_exporters():
    """Unit Test: Runs the exporters."""
    filepath = Path(__file__).parent / "test_output_file"

    # Generate a 2D single material mock result
    (nelx_2d, nely_2d, nelz_2d) = 30, 20, 0
    nel_2d = nelx_2d * nely_2d
    p = (
        1 / 0.3 - 1
    )  # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel_2d)
    densities_2d = x**p
    np.random.shuffle(densities_2d)
    result_2d = densities_2d

    # Generate a 3D single material mock result
    nelx_3d, nely_3d, nelz_3d = 30, 20, 10
    nel_3d = nelx_3d * nely_3d * nelz_3d
    x = np.linspace(0, 1, nel_3d)
    densities_3d = x**p
    np.random.shuffle(densities_3d)
    result_3d = densities_3d

    # Generate Multi-material mock results
    result_2d_multi = np.vstack([result_2d * 0.5, result_2d * 0.3])
    result_3d_multi = np.vstack([result_3d * 0.5, result_3d * 0.3])
    colors = ["#FF0000", "#00FF00"]

    png_path = str(Path(__file__).parent / "test_output_file.png")

    for res, dims, is_multi in [
        (result_2d, (nelx_2d, nely_2d, nelz_2d), False),
        (result_3d, (nelx_3d, nely_3d, nelz_3d), False),
        (result_2d_multi, (nelx_2d, nely_2d, nelz_2d), True),
        (result_3d_multi, (nelx_3d, nely_3d, nelz_3d), True),
    ]:
        args = (res, dims, png_path) if not is_multi else (res, dims, png_path, colors)
        success, error_msg = exporters.save_as_png(*args)
        assert success, f"PNG export failed: {error_msg}"

        args = (res, dims, str(filepath) + ".vti")
        success, error_msg = exporters.save_as_vti(*args)
        assert success, f"VTI export failed: {error_msg}"

        args = (res, dims, str(filepath) + ".stl")
        success, error_msg = exporters.save_as_stl(*args)
        assert success, f"STL export failed: {error_msg}"

        threemf_path = str(Path(__file__).parent / "test_output_file.3mf")
        args = (
            (res, dims, threemf_path)
            if not is_multi
            else (res, dims, threemf_path, colors)
        )
        success, error_msg = exporters.save_as_3mf(*args)
        assert success, f"3MF export failed: {error_msg}"


def test_stl_2d_lies_in_the_xy_plane(tmp_path):
    """A 2D design is extruded along z, not rotated into the yz plane."""
    from stl import mesh

    nx, ny = 10, 4
    densities = np.ones(nx * ny)
    path = tmp_path / "block.stl"
    success, error = exporters.save_as_stl(densities, [nx, ny, 0], str(path))
    assert success, error

    loaded = mesh.Mesh.from_file(str(path))
    low, high = loaded.min_, loaded.max_
    np.testing.assert_allclose(low, [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(high, [nx, ny, 1.0], atol=1e-5)


def test_3mf_2d_multi_material_colours(tmp_path):
    """Each triangle of a 2D multi-material part gets its material colour."""
    if exporters.lib3mf is None:  # pragma: no cover - platform without lib3mf
        import pytest

        pytest.skip("lib3mf is not installed")

    import lib3mf

    nx, ny = 20, 6
    densities = np.zeros((2, nx * ny))
    element_x = np.repeat(np.arange(nx), ny)
    densities[0, element_x < 10] = 1.0
    densities[1, element_x >= 10] = 1.0
    path = tmp_path / "split.3mf"
    success, error = exporters.save_as_3mf(
        densities, [nx, ny, 0], str(path), ["#FF0000", "#0000FF"]
    )
    assert success, error

    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    model.QueryReader("3mf").ReadFromFile(str(path))
    iterator = model.GetMeshObjects()
    iterator.MoveNext()
    mesh_object = iterator.GetCurrentMeshObject()
    vertices = mesh_object.GetVertices()
    triangles = mesh_object.GetTriangleIndices()

    ids = {"left": set(), "right": set()}
    for index in range(mesh_object.GetTriangleCount()):
        prop = mesh_object.GetTriangleProperties(index).PropertyIDs[0]
        centre = np.mean(
            [list(vertices[k].Coordinates) for k in triangles[index].Indices], axis=0
        )
        side = "left" if centre[0] < 10 else "right"
        ids[side].add(prop)

    assert ids["left"] == {1}
    assert ids["right"] == {2}


def test_stl_keeps_diagonally_touching_voxels_connected(tmp_path):
    """Two solid voxels sharing only an edge must export as one shell.

    Regression: contouring the solid field with VTK's case table separated
    such voxels, so a connected design was exported as many disjoint shells.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from stl import mesh

    nx, ny, nz = 2, 2, 1
    densities = np.zeros(nx * ny * nz)
    densities[0 * ny + 0] = 1.0  # element (0, 0, 0)
    densities[1 * ny + 1] = 1.0  # element (1, 1, 0): touches only along an edge
    path = tmp_path / "diagonal.stl"
    success, error = exporters.save_as_stl(densities, [nx, ny, nz], str(path))
    assert success, error

    vectors = mesh.Mesh.from_file(str(path)).vectors.reshape(-1, 3)
    _, vertex = np.unique(np.round(vectors, 5), axis=0, return_inverse=True)
    triangles = vertex.reshape(-1, 3)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]]])
    n = int(triangles.max()) + 1
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    assert connected_components(graph)[0] == 1

    signed = mesh.Mesh.from_file(str(path)).vectors.astype(np.float64)
    volume = (
        np.einsum("ij,ij->i", signed[:, 0], np.cross(signed[:, 1], signed[:, 2])).sum()
        / 6
    )
    assert volume > 0, "normals must point outwards"
