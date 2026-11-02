"""
Fixed-point diagnostics for the Picard map P of gs_solver.py: relative residuals and dense Jacobian eigenvalues.
"""
import numpy as np
import gs_solver as gs


def make_map(A, bnd, rmesh, coords):
    """Return the one-step Picard map P(psi) for an already-assembled problem."""
    return lambda x: gs._picard_step(x, A, rmesh, coords, bnd)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def residuals(P, x):
    """(|P x - x|, |P(P x) - x|), both relative.  A true solution has the first one ~ 0."""
    return rel(P(x), x), rel(P(P(x)), x)


def dense_jacobian(P, x, eps_rel=1e-6):
    """Central-difference dense Jacobian L = dP/dpsi (use on small grids, N <~ 1500)."""
    N = x.size
    L = np.zeros((N, N)); e0 = eps_rel * np.linalg.norm(x)
    for j in range(N):
        v = np.zeros(N); v[j] = 1.0
        L[:, j] = (P(x + e0 * v) - P(x - e0 * v)) / (2 * e0)
    return L


def sorted_eigs(M):
    lam = np.linalg.eigvals(M)
    return lam[np.argsort(-np.abs(lam))]
