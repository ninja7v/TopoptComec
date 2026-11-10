# topoptcomec/core/direct_solver.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Sparse direct factorisation of symmetric stiffness matrices.

from __future__ import annotations

import warnings
import weakref
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, csr_matrix, triu
from scipy.sparse.linalg import splu

try:  # Optional accelerator: Intel MKL PARDISO (``pip install pypardiso``).
    import pypardiso
except ImportError:  # pragma: no cover - depends on the installation
    pypardiso = None

# Type aliases
FloatArray = npt.NDArray[np.float64]
SolveFunction = Callable[[FloatArray], FloatArray]

#: SuperLU diagonal pivot threshold used in symmetric mode: the diagonal entry
#: is kept as pivot unless it is this much smaller than the column maximum.
DIAGONAL_PIVOT_THRESHOLD = 1e-3
#: Iterative refinement steps applied to the SuperLU solutions.
REFINEMENT_STEPS = 2
#: PARDISO matrix type: real symmetric positive definite (Cholesky).
PARDISO_POSITIVE_DEFINITE = 2
#: Matrices with fewer stored entries than this are factorised with SuperLU
#: even when PARDISO is available: its thread pool only pays off on larger
#: systems (and slows down the surrounding NumPy code on small ones).
PARDISO_MIN_ENTRIES = 500_000


def pardiso_available() -> bool:
    """Whether the optional MKL PARDISO backend can be used."""
    return pypardiso is not None


class _Pardiso:
    """
    Cholesky factorisation with MKL PARDISO.

    The symbolic analysis (fill-reducing ordering) only depends on the
    sparsity pattern, which is fixed during an optimization: it is done once
    and every later call only repeats the numerical factorisation.
    """

    def __init__(self) -> None:
        self._solver = pypardiso.PyPardisoSolver(mtype=PARDISO_POSITIVE_DEFINITE)
        self._pattern: tuple[int, int] | None = None
        # PARDISO keeps its factors in native memory until told otherwise.
        weakref.finalize(self, self._solver.free_memory, everything=True)

    def _run(self, phase: int, matrix: csr_matrix, rhs: FloatArray) -> FloatArray:
        self._solver.set_phase(phase)
        return self._solver._call_pardiso(matrix, rhs)

    def factorize(self, K: csc_matrix) -> SolveFunction:
        """Factorise ``K`` (symmetric positive definite)."""
        upper = triu(K, format="csr")
        upper.sort_indices()
        self._solver._check_A(upper)
        empty = np.zeros((upper.shape[0], 1))
        pattern = (upper.shape[0], upper.nnz)
        if pattern != self._pattern:
            self._pattern = None
            self._run(11, upper, empty)  # symbolic analysis
            self._pattern = pattern
        try:
            self._run(22, upper, empty)  # numerical factorisation
        except Exception:
            self._pattern = None
            raise

        def solve(rhs: FloatArray) -> FloatArray:
            columns = np.asfortranarray(rhs, dtype=np.float64)
            return self._run(33, upper, columns).reshape(np.shape(rhs))

        return solve


class DirectSolver:
    """
    Sparse direct solver for symmetric stiffness (or tangent) matrices.

    When the optional ``pypardiso`` package is installed, large systems are
    factorised with MKL PARDISO: it is multi-threaded and several times (2D)
    to tens of times (3D) faster than SuperLU.

    Otherwise - or when PARDISO fails, e.g. on an indefinite tangent matrix -
    SuperLU runs in symmetric mode (minimum-degree ordering of ``A + A^T``,
    diagonal pivots preferred). On stiffness matrices this roughly halves the
    fill-in and the factorisation time compared with the default unsymmetric
    column ordering. If that factorisation breaks down too, the default
    partial-pivoting one is used.
    """

    def __init__(self) -> None:
        self._pardiso: _Pardiso | None = None

    def factorize(self, K: csc_matrix) -> SolveFunction:
        """
        Factorise ``K``.

        Parameters
        ----------
        K : csc_matrix
            Symmetric matrix.

        Returns
        -------
        SolveFunction
            Solve function accepting a vector or a matrix of right-hand
            sides. It is valid until the next call of :meth:`factorize`.

        Raises
        ------
        RuntimeError
            If the matrix is singular.
        """
        solve = self._factorize_pardiso(K) if self._use_pardiso(K) else None
        if solve is None:
            solve = self._factorize_superlu(K)

        def checked(rhs: FloatArray) -> FloatArray:
            solution = np.asarray(solve(rhs), dtype=np.float64)
            if not np.all(np.isfinite(solution)):
                raise RuntimeError("the factorisation produced non-finite values")
            return solution

        return checked

    @staticmethod
    def _use_pardiso(K: csc_matrix) -> bool:
        return pypardiso is not None and K.nnz >= PARDISO_MIN_ENTRIES

    def _factorize_pardiso(self, K: csc_matrix) -> SolveFunction | None:
        """PARDISO factorisation, or ``None`` when it fails."""
        try:
            if self._pardiso is None:
                self._pardiso = _Pardiso()
            return self._pardiso.factorize(K)
        except Exception as error:  # noqa: BLE001 - any failure -> SuperLU
            # A PARDISO error is expected for an indefinite (tangent) matrix,
            # which SuperLU handles; anything else deserves a notice.
            if type(error).__name__ != "PyPardisoError":
                warnings.warn(
                    f"PARDISO failed ({error}); using SuperLU instead.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return None

    @staticmethod
    def _factorize_superlu(K: csc_matrix) -> SolveFunction:
        try:
            factors = splu(
                K,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=DIAGONAL_PIVOT_THRESHOLD,
                options={"SymmetricMode": True},
            )
        except RuntimeError:
            factors = splu(K)

        def solve(rhs: FloatArray) -> FloatArray:
            # The weak pivoting of the symmetric mode costs a few digits;
            # iterative refinement restores them for one extra cheap solve.
            solution = factors.solve(rhs)
            for _ in range(REFINEMENT_STEPS):
                solution = solution + factors.solve(rhs - K @ solution)
            return solution

        return solve
