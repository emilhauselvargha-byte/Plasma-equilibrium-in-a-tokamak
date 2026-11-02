#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════════════════
         Grad–Shafranov Solver  —  Unified Configuration                    
                                                                              
  QUICK-START GUIDE                                                           
  ─────────────────                                                           
  1. Pick a GEOMETRY (Section I-A):   SHAFRANOV  or  SOLOVIEV                
  2. Pick a PHYSICS MODE (Section I-B): SOLOVIEV_LINEAR, PICARD_NONLINEAR,  
                                        or CUSTOM                             
  3. Set the matching parameters (Section I-C / I-D).                        
  4. Run.                                                                     
                                                                              
  GEOMETRY controls the plasma cross-section shape.                           
  PHYSICS MODE controls what p(ψ), F(ψ) profiles you use.                    
  Any geometry can be combined with any physics mode.                         
══════════════════════════════════════════════════════════════════════════════

PHYSICS BACKGROUND
──────────────────
The Grad–Shafranov equation is:

    Δ*ψ  =  −μ₀ R² dp/dψ  −  F dF/dψ  =  S(R, ψ)

where Δ*ψ = ∂²ψ/∂R² − (1/R)∂ψ/∂R + ∂²ψ/∂Z²  (Grad–Shafranov operator).

Three ready-made physics modes:

  SOLOVIEV_LINEAR   — Exact analytic solution exists. Solved by a single
      linear system (fast). Source is S(R,ψ) = (8b+2)R²/a = const × R².
      Profiles: p(ψ) = P_sol·ψ,  F²(ψ) = F0² + F_sol·ψ.
      Only available with SOLOVIEV geometry (boundary and source are coupled).

  PICARD_NONLINEAR  — Iterative (Picard) solve for arbitrary p(ψ), F(ψ), with the profile shapes applied
      to −p'(ψ_n) and −(F²)'(ψ_n) directly (source nonzero on the axis). Section I-D.
      Can be used with either SHAFRANOV or SOLOVIEV geometry. Plain Picard iteration.

  CUSTOM            — Drop in your own source function S(R, psi_normalised)
      in Section I-D. Full control.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import bisect, brentq, newton, minimize_scalar
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, LinearOperator, eigs
import time

# Output / display controls
OUTPUT_DIR = 'figures'
THEME = 'light'              # 'light' (white background, default) or 'dark'
SHOW = False                # True for interactive windows (e.g. %matplotlib qt)
SPECTRAL_ANALYSIS = False   # Arnoldi study of the Picard Jacobian (slow on fine grids)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION I-A  ·  GEOMETRY SELECTION
# ══════════════════════════════════════════════════════════════════════════════
#
#  GEOMETRY = 'SHAFRANOV'  — classic Miller / Shafranov parametric boundary
#                            t ∈ [0, 2π]:  R(t) = R0 + a·cos(t + δ·sin(t))
#                                           Z(t) = κ·a·sin(t)
#
#  GEOMETRY = 'SOLOVIEV'   — boundary given analytically by
#                            Z(R) = ±√(a − b(R²−R0²)²) / R
#                            (closed D-shape; requires R0, k0, q0 — see I-C)
#
#GEOMETRY = 'SOLOVIEV'           # <── change here
GEOMETRY = 'SHAFRANOV'

# ══════════════════════════════════════════════════════════════════════════════
# SECTION I-B  ·  PHYSICS MODE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
#
#  PHYSICS = 'SOLOVIEV_LINEAR'   — single linear solve, exact analytic reference
#                                   (only valid with GEOMETRY = 'SOLOVIEV')
#
#  PHYSICS = 'PICARD_NONLINEAR'  — iterative Picard solve with the shape
#                                   functions defined in Section I-D
#
#  PHYSICS = 'CUSTOM'            — you supply source_custom(r, psi_n) directly
#
#PHYSICS = 'SOLOVIEV_LINEAR'     # <── change here
PHYSICS = 'PICARD_NONLINEAR'
#PHYSICS = 'CUSTOM'

# ══════════════════════════════════════════════════════════════════════════════
# SECTION I-C  ·  GEOMETRY PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared ────────────────────────────────────────────────────────────────────
R0  = 0.95     # major radius (normalised units; set to 1 for true normalisation)
ds  = 5e-2    # arc-length step along boundary (smaller = finer grid, slower)

# ── SHAFRANOV shape parameters (ignored when GEOMETRY = 'SOLOVIEV') ──────────
a_sha     = 0.6   # minor radius  (normalised to R0 if R0=1, else metres)
delta_sha = 0.8   # triangularity
kappa_sha = 1   # elongation

# ── SOLOVIEV shape parameters (ignored when GEOMETRY = 'SHAFRANOV') ──────────
#    These also fix the Soloviev source when PHYSICS = 'SOLOVIEV_LINEAR'.
k0    = 2.2   # elongation parameter  (controls Z-extent)
q0    = 1.1   # safety-factor parameter (controls R-extent and source amplitude)
B0    = 0.5   # reference field (absorbed into normalisation)
psi_b = 0.09   # Dirichlet boundary value of ψ

# ══════════════════════════════════════════════════════════════════════════════
# SECTION I-D  ·  PHYSICS PROFILES  (PICARD_NONLINEAR and CUSTOM)
# ══════════════════════════════════════════════════════════════════════════════
#
# psi_n = (ψ − ψ_min) / (ψ_bnd − ψ_min) ∈ [0, 1]   (normalised flux)
#
# ── Profile shapes: applied to the DERIVATIVES p'(ψ) and FF'(ψ) ─────────────────
#    (so that the source, hence the toroidal current density, is NONZERO on the magnetic axis)
#
#       −dp/dψ_n      =  p0    · (1 − ψ_n^sp1)^sp2
#       −d(F²)/dψ_n   =  alpha · (1 − ψ_n^sf1)^sf2          ( −FF' = −½ d(F²)/dψ_n )
#
#    Both shapes equal 1 on the axis (ψ_n = 0) and 0 on the boundary (ψ_n = 1).
#    Source:   S = −R² dp/dψ_n − ½ d(F²)/dψ_n  =  R² p0 · p_shape + (alpha/2) · f2_shape  ≥ 0
#
#    NOTE: only the ratio p0/alpha matters for the normalised Picard map (the overall amplitude is removed
#    by the ψ_n renormalisation).
sp1 = 2
sp2 = 2
sf1 = 2
sf2 = 1

# ── Picard amplitude parameters ───────────────────────────────────────────────
p0    = 1   # pressure-gradient amplitude   (scales the R² term of S)
alpha = 1   # FF' amplitude                  (scales the constant term of S)

# ── Picard iteration controls ────────────────────────────────────────────────
#  Plain Picard  ψ ← P(ψ).  Stopping test: fixed-point residual ‖P(ψ) − ψ‖/‖ψ‖ < PICARD_EPS.
MAX_ITER      = 300
PICARD_EPS    = 1e-9
LAST_DIAG     = {}      # filled by solve_picard: iterations, res_P, res_P2

# ── Custom source (used only when PHYSICS = 'CUSTOM') ────────────────────────
def source_custom(r, psi_n):
    """
    User-defined RHS of Δ*ψ = source_custom(R, ψ_n).

    Parameters
    ----------
    r     : R coordinate of the node
    psi_n : normalised flux at that node ∈ [0, 1]

    Returns  S(R, ψ_n)

    Example below: Soloviev-like linear source (replace freely).
    """
    # Example: constant current + pressure gradient
    #   S = -R² * (dp/dψ) - (F dF/dψ)
    #   dp/dψ = -p0  →  pressure contribution: p0 * R²
    #   F dF/dψ = -alpha/2  →  current contribution: alpha/2
    return p0 * r**2 + alpha / 2.0

# ══════════════════════════════════════════════════════════════════════════════
# SECTION II  ·  DERIVED SOLOVIEV CONSTANTS  (auto-computed, do not edit)
# ══════════════════════════════════════════════════════════════════════════════

a_sol = 2.0 * R0**2 * k0 * q0 * psi_b / B0   # a = 2R0²k0q0ψb/B0
b_sol = k0**2 / 4.0                            # b = k0²/4

# ══════════════════════════════════════════════════════════════════════════════
# SECTION III  ·  BOUNDARY GEOMETRY IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ─── A. Shafranov (parametric) boundary ──────────────────────────────────────

def _Rb_sha(t):
    """R on Shafranov boundary at parameter t ∈ [0, 2π]."""
    return (R0 + a_sha * np.cos(t + np.pi + delta_sha * np.sin(t+np.pi))) / R0

def _Zb_sha(t):
    """Z on Shafranov boundary at parameter t ∈ [0, 2π]."""
    return kappa_sha * a_sha * np.sin(t) / R0

def _dRb_sha(t, h=1e-7):
    return (_Rb_sha(t + h) - _Rb_sha(t - h)) / (2 * h)

def _dZb_sha(t, h=1e-7):
    return (_Zb_sha(t + h) - _Zb_sha(t - h)) / (2 * h)

def _arc_speed_sha(t):
    return np.sqrt(_dRb_sha(t)**2 + _dZb_sha(t)**2)

def sample_outer_boundary_sha():
    """
    Walk from t=π (outer midplane, Z=0) to t=π/2 (top) along the Shafranov
    boundary, using constant arc-length steps ds.
    Returns rs, zs with Z ascending (z=0 first).
    """
    print("  Sampling Shafranov outer boundary (t: π → π/2) …")
    t     = np.pi
    t_end = np.pi / 2.0
    rs, zs = [_Rb_sha(t)], [_Zb_sha(t)]

    while True:
        r0, z0 = rs[-1], zs[-1]
        dt_est = ds / _arc_speed_sha(t)

        def f(tau):
            return (_Rb_sha(tau) - r0)**2 + (_Zb_sha(tau) - z0)**2 - ds**2

        t_lo = t - 1.3 * dt_est
        t_hi = t - 0.2 * dt_est

        if t_lo <= t_end:
            break
        # try:
        #     t_new = bisect(f, t_lo, t_hi, xtol=1e-11)
        # except ValueError:
        try:
            t_new = newton(f, t - dt_est)
        except Exception:
            try:
                t_new = brentq(f, t_lo, t_hi)
            except Exception:
                break

        if np.pi <= t_new or t_new <= t_end or t < t_new:
            break

        t = t_new
        rs.append(_Rb_sha(t))
        zs.append(_Zb_sha(t))

    rs.append(_Rb_sha(t_end))
    zs.append(_Zb_sha(t_end))
    rs, zs = np.array(rs), np.array(zs)
    print(f"    {len(rs)} pts | R(z=0)={rs[0]:.4f}  R(top)={rs[-1]:.4f}  Z_max={zs[-1]:.4f}")
    return rs, zs

def find_inner_r_at_z_sha(z_target):
    """R on inner (HFS) Shafranov boundary at height z_target ≥ 0."""
    if z_target <= 1e-12:
        return _Rb_sha(0.0)
    def f(t):
        return _Zb_sha(t) - z_target
    try:
        t_sol = bisect(f, 0.0, np.pi / 2.0, xtol=1e-12)
    except ValueError:
        try:
            t_sol = newton(f, np.pi / 4)
        except Exception:
            try:
                t_sol = brentq(f, 0, np.pi / 2)
            except Exception:
                t_sol = 0.0 if abs(f(0.0)) < abs(f(np.pi / 2.0)) else np.pi / 2.0
    return _Rb_sha(t_sol)

def boundary_curve_sha():
    """Full parametric boundary curve for plotting (t ∈ [0, 2π])."""
    ts = np.linspace(0, 2 * np.pi, 800)
    return _Rb_sha(ts), _Zb_sha(ts)

# ─── B. Soloviev (analytic) boundary ─────────────────────────────────────────

def _Zbnd_sq_sol(R):
    """Z²(R) on Soloviev boundary: (a − b(R²−R0²)²)/ R²."""
    return (a_sol - b_sol * (R**2 - R0**2)**2) / R**2

def Z_boundary_sol(R):
    """Upper Soloviev boundary Z(R) ≥ 0; returns NaN outside plasma."""
    val = a_sol - b_sol * (R**2 - R0**2)**2
    if np.isscalar(R):
        return np.sqrt(max(val, 0.0)) / R if val >= 0 else np.nan
    val = np.asarray(val, float)
    out = np.full_like(val, np.nan)
    m = val >= 0
    out[m] = np.sqrt(val[m]) / np.asarray(R)[m]
    return out

def _compute_soloviev_limits():
    """Precompute R_min, R_max, R_Zmax, Z_max for Soloviev boundary."""
    sqrt_ab = np.sqrt(a_sol / b_sol)
    R_max = np.sqrt(R0**2 + sqrt_ab)
    R_min = np.sqrt(max(R0**2 - sqrt_ab, 1e-8))
    res   = minimize_scalar(lambda R: -_Zbnd_sq_sol(R), bounds=(R_min, R_max), method='bounded')
    R_Zmax = res.x
    Z_max  = np.sqrt(-res.fun)
    return R_min, R_max, R_Zmax, Z_max

# Compute once at import time
_R_min_sol, _R_max_sol, _R_Zmax_sol, _Z_max_sol = _compute_soloviev_limits()

def sample_outer_boundary_sol():
    """
    Walk from R_max (Z=0) to R_Zmax (Z=Z_max) along the Soloviev outer boundary.
    Returns rs, zs with Z ascending.
    """
    print("  Sampling Soloviev outer boundary (R: R_max → R_Zmax) …")
    R_max, R_Zmax = _R_max_sol, _R_Zmax_sol
    Rs = [R_max]
    Zs = [0.0]
    R  = R_max

    while True:
        R0_loc, Z0_loc = Rs[-1], Zs[-1]

        def dist_sq(R_new):
            v = _Zbnd_sq_sol(R_new)
            Z_new = np.sqrt(max(v, 0.0)) if v >= 0 else 0.0
            return (R_new - R0_loc)**2 + (Z_new - Z0_loc)**2 - ds**2

        R_lo = max(R_Zmax, R - 2 * ds)
        R_hi = R - 1e-8

        if R_lo >= R_hi or R <= R_Zmax + 1e-8:
            break
        try:
            if dist_sq(R_lo) * dist_sq(R_hi) > 0:
                R_new = R - ds * 0.8
            else:
                R_new = brentq(dist_sq, R_lo, R_hi, xtol=1e-10)
        except Exception:
            R_new = R - ds * 0.8

        if R_new <= R_Zmax:
            break
        R = R_new
        Rs.append(R)
        Zs.append(np.sqrt(max(_Zbnd_sq_sol(R), 0.0)))

    Rs.append(R_Zmax)
    Zs.append(_Z_max_sol)
    Rs, Zs = np.array(Rs), np.array(Zs)
    print(f"    {len(Rs)} pts | Z range: [{Zs[0]:.4f}, {Zs[-1]:.4f}]")
    return Rs, Zs

def find_inner_r_at_z_sol(z_target):
    """R on inner Soloviev boundary at height z_target (R in [R_min, R_Zmax])."""
    R_min, R_Zmax = _R_min_sol, _R_Zmax_sol
    if z_target <= 1e-12:
        return R_min
    if z_target >= _Z_max_sol - 1e-12:
        return R_Zmax
    def f(R):
        v = _Zbnd_sq_sol(R)
        return (np.sqrt(max(v, 0.0)) if v >= 0 else 0.0) - z_target
    try:
        return brentq(f, R_min + 1e-8, R_Zmax - 1e-8, xtol=1e-12)
    except Exception:
        Rs = np.linspace(R_min, R_Zmax, 500)
        fv = np.array([f(r) for r in Rs])
        idx = np.where(np.diff(np.sign(fv)))[0]
        if len(idx):
            return brentq(f, Rs[idx[0]], Rs[idx[0] + 1], xtol=1e-12)
        return R_min

def boundary_curve_sol():
    """Full Soloviev boundary curve for plotting."""
    Rs = np.linspace(_R_min_sol, _R_max_sol, 1000)
    Zp = np.array([Z_boundary_sol(r) if _Zbnd_sq_sol(r) >= 0 else np.nan for r in Rs])
    return Rs, np.concatenate([-Zp[::-1], Zp])  # (Rs mirrored, full Z)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION IV  ·  GEOMETRY DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def sample_outer_boundary():
    if GEOMETRY == 'SHAFRANOV':
        return sample_outer_boundary_sha()
    return sample_outer_boundary_sol()

def find_inner_r_at_z(z_target):
    if GEOMETRY == 'SHAFRANOV':
        return find_inner_r_at_z_sha(z_target)
    return find_inner_r_at_z_sol(z_target)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION V  ·  GRID CONSTRUCTION  (shared)
# ══════════════════════════════════════════════════════════════════════════════

def build_grid(rs_outer, zs):
    print("  Building upper-half grid …")
    rs_inner = np.array([find_inner_r_at_z(z) for z in zs])
    zmesh = np.array(zs)
    rmesh = np.unique(np.round(np.concatenate([rs_inner, rs_outer]), 10))
    Nr, Nz = len(rmesh), len(zmesh)
    r_inner_row = rs_inner.copy()
    r_outer_row = rs_outer.copy()

    inside = np.zeros((Nr, Nz), dtype=bool)
    for iz in range(Nz):
        ri, ro = r_inner_row[iz], r_outer_row[iz]
        for ir in range(Nr):
            if ri - 1e-9 <= rmesh[ir] <= ro + 1e-9:
                inside[ir, iz] = True

    print(f"    Nr={Nr}, Nz={Nz}, active={inside.sum()}")
    print(f"    R ∈ [{rmesh[0]:.4f}, {rmesh[-1]:.4f}],  Z ∈ [{zmesh[0]:.4f}, {zmesh[-1]:.4f}]")
    return rmesh, zmesh, inside, r_inner_row, r_outer_row

def build_index_map(inside):
    Nr, Nz = inside.shape
    node_id = -np.ones((Nr, Nz), dtype=int)
    coords  = []
    k = 0
    for iz in range(Nz):
        for ir in range(Nr):
            if inside[ir, iz]:
                node_id[ir, iz] = k
                coords.append((ir, iz))
                k += 1
    return node_id, coords

# ══════════════════════════════════════════════════════════════════════════════
# SECTION VI  ·  FD STENCIL / MATRIX ASSEMBLY  (shared)
# ══════════════════════════════════════════════════════════════════════════════

def is_dirichlet(ir, iz, inside, Nz):
    """
    Node is Dirichlet (ψ = ψ_bnd) if it lies on the plasma boundary, i.e.:
      · top row (iz = Nz−1), OR
      · missing / inactive R-neighbour (inner or outer wall), OR
      · interior rows: missing Z-neighbour (upper/lower wall).
    Bottom row (iz=0) is NOT Dirichlet — it gets the Neumann (symmetry) BC.
    """
    Nr = inside.shape[0]
    if iz == Nz - 1:                                      return True
    if ir == 0 or ir == Nr - 1:                           return True
    if not inside[ir + 1, iz] or not inside[ir - 1, iz]: return True
    if iz > 0:
        if not inside[ir, iz + 1] or not inside[ir, iz - 1]: return True
    if iz == 0 and not inside[ir, iz + 1]:                return True
    return False

def assemble(rmesh, zmesh, inside, node_id, coords):
    """
    Assemble sparse matrix for Δ*ψ using non-uniform finite differences.

    Δ*ψ = ∂²ψ/∂R² − (1/R)∂ψ/∂R + ∂²ψ/∂Z²

    At iz=0 (midplane): ghost-point Neumann (∂ψ/∂Z = 0) →
        ∂²ψ/∂Z²|₀ ≈ 2(ψ[iz+1] − ψ[iz]) / dz_p²
    """
    N  = len(coords)
    Nr, Nz = inside.shape
    A  = lil_matrix((N, N))
    bnd = set()

    for k, (ir, iz) in enumerate(coords):
        if is_dirichlet(ir, iz, inside, Nz):
            A[k, k] = 1.0
            bnd.add(k)
            continue

        r    = rmesh[ir]
        dr_p = rmesh[ir + 1] - rmesh[ir]
        dr_m = rmesh[ir]     - rmesh[ir - 1]
        dr_s = dr_p + dr_m

        # Radial stencil: ∂²/∂R² − (1/R)∂/∂R
        c_rp =  2.0/(dr_p*dr_s) - dr_m**2 / (r * dr_p * dr_m * dr_s)
        c_rm =  2.0/(dr_m*dr_s) + dr_p**2 / (r * dr_p * dr_m * dr_s)
        c_r0 = -2.0/(dr_p*dr_m) + (dr_m**2 - dr_p**2) / (r * dr_p * dr_m * dr_s)

        k_rp = node_id[ir + 1, iz]
        k_rm = node_id[ir - 1, iz]

        dz_p = zmesh[iz + 1] - zmesh[iz]

        if iz == 0:   # ghost-point Neumann  ∂ψ/∂Z = 0
            c_zp, c_zm, c_z0 = 2.0/dz_p**2, 0.0, -2.0/dz_p**2
            k_zp, k_zm = node_id[ir, iz + 1], -1
        else:
            dz_m = zmesh[iz] - zmesh[iz - 1]
            dz_s = dz_p + dz_m
            c_zp =  2.0 / (dz_p * dz_s)
            c_zm =  2.0 / (dz_m * dz_s)
            c_z0 = -2.0 / (dz_p * dz_m)
            k_zp, k_zm = node_id[ir, iz + 1], node_id[ir, iz - 1]

        A[k, k]    = c_r0 + c_z0
        A[k, k_rp] = c_rp
        A[k, k_rm] = c_rm
        A[k, k_zp] += c_zp
        if k_zm >= 0:
            A[k, k_zm] = c_zm

    print(f"    Dirichlet nodes: {len(bnd)}  /  {N} DOFs total")
    return A.tocsr(), bnd

# ══════════════════════════════════════════════════════════════════════════════
# SECTION VII  ·  PHYSICS SOURCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ─── A. Soloviev linear source ────────────────────────────────────────────────

def source_soloviev(r, psi_n=None):
    """
    Exact source for ψ = [b(R²−R0²)² + R²Z²] / a:
        Δ*ψ = (8b + 2) R² / a
    psi_n is unused (linear system — no dependence on ψ).
    """
    return (8.0 * b_sol + 2.0) * r**2 / a_sol

def psi_analytic_soloviev(R, Z):
    """
    Soloviev analytic solution (un-normalised physical units):
        ψ(R,Z) = psi_b · [b(R²−R0²)² + R²Z²] / a
    Equals psi_b on the boundary by construction.
    """
    return psi_b * (b_sol * (R**2 - R0**2)**2 + R**2 * Z**2) / a_sol

# ─── B. Picard nonlinear profiles ────────────────────────────────────────────

def p_shape(pn):
    """Shape of −dp/dψ_n  (= 1 on axis, 0 on boundary)."""
    return (1.0 - np.clip(pn, 0.0, 1.0)**sp1)**sp2

def f2_shape(pn):
    """Shape of −d(F²)/dψ_n  (= 1 on axis, 0 on boundary)."""
    return (1.0 - np.clip(pn, 0.0, 1.0)**sf1)**sf2

def source_picard(r, pn):
    """
    RHS for Picard iteration, profiles shaped directly on the derivatives:
        S = −R² dp/dψ_n − ½ d(F²)/dψ_n = R² p0 · p_shape(ψ_n) + (α/2) · f2_shape(ψ_n)  ≥ 0
    (S > 0 so that ψ has its minimum — the magnetic axis — inside the plasma.)
    """
    return r**2 * p0 * p_shape(pn) + (alpha / 2.0) * f2_shape(pn)

def _deriv(f, pn, h=1e-5):
    """Clipped central difference (helper for CUSTOM sources defined through p and F^2 themselves)."""
    lo = np.clip(pn - h, 0.0, 1.0)
    hi = np.clip(pn + h, 0.0, 1.0)
    return (f(hi) - f(lo)) / (hi - lo + 1e-30)

# ─── C. Dispatch ─────────────────────────────────────────────────────────────

def get_source_fn():
    """Return the source function S(r, psi_n) for the selected PHYSICS mode."""
    if PHYSICS == 'SOLOVIEV_LINEAR':
        return source_soloviev
    elif PHYSICS == 'PICARD_NONLINEAR':
        return source_picard
    elif PHYSICS == 'CUSTOM':
        return source_custom
    else:
        raise ValueError(f"Unknown PHYSICS mode: '{PHYSICS}'")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION VIII  ·  SOLVERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_psi(psi, bnd):
    """Shift/scale so min→0, boundary-mean→1."""
    psi_min = psi.min()
    psi_bnd = np.mean([psi[k] for k in bnd])
    denom   = psi_bnd - psi_min
    if abs(denom) < 1e-14:
        return psi
    return (psi - psi_min) / denom

def solve_linear(A_csr, bnd, rmesh, coords):
    """
    Single linear solve.  Used for SOLOVIEV_LINEAR (and CUSTOM if source
    has no ψ-dependence).
    """
    source_fn = get_source_fn()
    N = len(coords)
    b_vec = np.zeros(N)
    for k, (ir, iz) in enumerate(coords):
        if k in bnd:
            b_vec[k] = 1
        else:
            b_vec[k] = source_fn(rmesh[ir], 0.0)   # psi_n unused for linear

    psi = spsolve(A_csr, b_vec)
    print(f"  ψ ∈ [{psi.min():.6f}, {psi.max():.6f}]")
    return psi*psi_b

def _picard_step(psi, A_csr, rmesh, coords, bnd):
    """One Picard step: evaluate source at current ψ, solve, re-normalise."""
    source_fn = get_source_fn()
    N = len(coords)
    b_vec = np.zeros(N)

    psi_min = psi.min()
    psi_bnd = np.mean([psi[k] for k in bnd])
    denom   = psi_bnd - psi_min
    if abs(denom) < 1e-14:
        denom = 1.0

    for k, (ir, iz) in enumerate(coords):
        if k in bnd:
            b_vec[k] = 1.0
        else:
            pn       = np.clip((psi[k] - psi_min) / denom, 0.0, 1.0)
            b_vec[k] = source_fn(rmesh[ir], pn)

    psi_new = spsolve(A_csr, b_vec)
    return _normalise_psi(psi_new, bnd)

def solve_picard(A_csr, bnd, rmesh, zmesh, coords):
    """
    Plain Picard iteration  ψ ← P(ψ)  for the nonlinear GS problem.
    The stopping test and the reported diagnostics use the true fixed-point residual ‖P(ψ) − ψ‖/‖ψ‖.
    """
    N = len(coords)
    P = lambda x: _picard_step(x, A_csr, rmesh, coords, bnd)
    rel = lambda a, b: np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-15)
    psi = np.zeros(N)
    for k in bnd:
        psi[k] = 1.0

    err = np.inf
    for it in range(MAX_ITER):
        psi_new = P(psi)
        err = rel(psi_new, psi)          # for plain Picard the step equals the residual
        psi = psi_new
        if it % 10 == 0:
            print(f"    iter {it:4d}   residual = {err:.3e}")
        if err < PICARD_EPS:
            break
    n_it = it + 1

    res_P  = rel(P(psi), psi)
    res_P2 = rel(P(P(psi)), psi)
    LAST_DIAG.clear(); LAST_DIAG.update(iterations=n_it, res_P=res_P, res_P2=res_P2)
    print(f"  {n_it} iterations   residual ‖Pψ−ψ‖/‖ψ‖ = {res_P:.2e}   ‖P²ψ−ψ‖/‖ψ‖ = {res_P2:.2e}")
    if res_P > 1e3 * PICARD_EPS:
        print("  ⚠ WARNING: not converged — ψ is NOT a fixed point of the Picard map.")
        if res_P2 < 1e-3 * res_P:
            print("    ‖P²ψ−ψ‖ is tiny: the iteration is oscillating on a period-2 cycle (see docs/paper, Sec. 4.3).")

    print(f"  ψ ∈ [{psi.min():.6f}, {psi.max():.6f}]")

    if not SPECTRAL_ANALYSIS:
        return psi*psi_b
    # SPECTRAL ANALYSIS of the Picard Jacobian L at the converged fixed point: plain Picard converges iff ρ(L) < 1.

    # 2. Matrix-free Jacobian-vector product (central difference)
    nrm = np.linalg.norm(psi)
    def Lv(v):
        eps = 1e-6 * nrm / np.linalg.norm(v)
        return (_picard_step(psi + eps * v, A_csr, rmesh, coords, bnd) - _picard_step(psi - eps * v, A_csr, rmesh, coords, bnd)) / (2 * eps)

    N = psi.size
    L_op = LinearOperator((N, N), matvec=Lv, dtype=float)

    # 3. Dominant eigenvalues of L
    lam = eigs(L_op, k=10, which="LM", return_eigenvectors=False)
    lam = lam[np.argsort(-np.abs(lam))]

    print(f"{'lambda':>24} {'|lambda|':>9}")
    for l in lam:
        print(f"{l:>24.4f} {abs(l):>9.4f}")
    print(f"rho(L) = {abs(lam).max():.4f}   (predicted iterations to 1e-11: {np.log(1e-11)/np.log(abs(lam).max()):.1f})")

    return psi*psi_b

# ══════════════════════════════════════════════════════════════════════════════
# SECTION IX  ·  MAIN SOLVE ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def solve():
    _validate_config()
    t0 = time.time()
    print("\n" + "═" * 62)
    print(f"  GS Solver  |  geometry={GEOMETRY}  |  physics={PHYSICS}")
    print("═" * 62)

    print("\n[1] Boundary sampling")
    rs_outer, zs = sample_outer_boundary()

    print("\n[2] Grid construction")
    rmesh, zmesh, inside, _, _ = build_grid(rs_outer, zs)
    node_id, coords = build_index_map(inside)

    print("\n[3] Matrix assembly")
    A_csr, bnd = assemble(rmesh, zmesh, inside, node_id, coords)

    print("\n[4] Solving")
    if PHYSICS == 'SOLOVIEV_LINEAR':
        psi = solve_linear(A_csr, bnd, rmesh, coords)
    else:
        psi = solve_picard(A_csr, bnd, rmesh, zmesh, coords)

    print(f"\n  Total time: {time.time() - t0:.2f}s")
    return psi, rmesh, zmesh, inside, node_id, coords, bnd

def _validate_config():
    if GEOMETRY == 'SHAFRANOV' and a_sha >= R0:
        raise ValueError(
            f"Shafranov boundary needs a_sha < R0 (got a_sha={a_sha}, R0={R0}): otherwise the plasma crosses "
            "R = 0, where the 1/R term of Δ* is singular. Reduce a_sha or increase R0."
        )
    if PHYSICS == 'SOLOVIEV_LINEAR' and GEOMETRY != 'SOLOVIEV':
        raise ValueError(
            "SOLOVIEV_LINEAR physics requires SOLOVIEV geometry.\n"
            "Either set GEOMETRY='SOLOVIEV' or switch to PICARD_NONLINEAR / CUSTOM."
        )
    # Check Soloviev domain is well-defined (R_min must be real)
    if GEOMETRY == 'SOLOVIEV':
        sqrt_ab = np.sqrt(a_sol / b_sol)
        if R0**4 <= sqrt_ab**2 + 1e-10:
            raise ValueError(
                f"Soloviev parameters give R_min² = R0²−√(a/b) ≤ 0.\n"
                f"  R0²={R0**2:.3f}, √(a/b)={sqrt_ab:.3f}\n"
                "  Reduce q0 or k0, or increase R0."
            )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION X  ·  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(psi, rmesh, zmesh, inside, coords, bnd, fname_override=None):
    Nr, Nz = inside.shape

    # Build upper-half 2D array
    psi_upper = np.full((Nr, Nz), np.nan)
    for k, (ir, iz) in enumerate(coords):
        psi_upper[ir, iz] = psi[k]

    # Mirror to full poloidal cross-section
    Z_full    = np.concatenate([-zmesh[::-1], zmesh[1:]])
    psi_full  = np.concatenate([psi_upper[:, ::-1], psi_upper[:, 1:]], axis=1)
    R_2d, Z_2d = np.meshgrid(rmesh, Z_full, indexing='ij')

    if THEME == 'dark':
        DARK, PANEL, FG, FG2, FG3, SPINE, CONT = '#0e0e14', '#13131f', 'white', '#aaa', '#ccc', '#444', 'white'
    else:   # light
        DARK, PANEL, FG, FG2, FG3, SPINE, CONT = 'white', 'white', 'black', '#333', '#222', '#888', 'white'

    # ── Decide what to plot ──────────────────────────────────────────────────
    has_analytic = (PHYSICS == 'SOLOVIEV_LINEAR' and GEOMETRY == 'SOLOVIEV')

    if has_analytic:
        # ── Build analytic solution on the same grid ──────────────────────
        psi_upper_ana = np.full((Nr, Nz), np.nan)
        for k, (ir, iz) in enumerate(coords):
            psi_upper_ana[ir, iz] = psi_analytic_soloviev(rmesh[ir], zmesh[iz])
        psi_full_ana = np.concatenate(
            [psi_upper_ana[:, ::-1], psi_upper_ana[:, 1:]], axis=1
        )
        psi_full_err = np.abs(psi_full - psi_full_ana)

        # ── Adaptive colour limits ────────────────────────────────────────
        # Use a shared range for the two ψ panels so they are directly
        # comparable, then a separate adaptive range for the error panel.
        psi_vmin = min(np.nanmin(psi_full), np.nanmin(psi_full_ana))
        psi_vmax = max(np.nanmax(psi_full), np.nanmax(psi_full_ana))
        err_max  = np.nanmax(psi_full_err)
        err_mean = np.nanmean(psi_full_err)

        datasets = [
            ('Numerical  ψ',    psi_full,     'plasma',  psi_vmin, psi_vmax, None),
            ('Analytic  ψ',     psi_full_ana, 'plasma',  psi_vmin, psi_vmax, None),
            (f'|ψ_num − ψ_ana|\n'
             f'max={err_max:.2e}   mean={err_mean:.2e}',
             psi_full_err, 'inferno', 0.0, err_max, 'sci'),
        ]
        ncols = 3
    else:
        # ── Adaptive colour limits for single-panel mode ──────────────────
        psi_vmin = np.nanmin(psi_full)
        psi_vmax = np.nanmax(psi_full)
        datasets = [
            ('ψ  equilibrium', psi_full, 'plasma', psi_vmin, psi_vmax, None),
        ]
        ncols = 1

    # ── Boundary curve ───────────────────────────────────────────────────────
    if GEOMETRY == 'SHAFRANOV':
        Rb_plot, Zb_plot = boundary_curve_sha()
    else:
        Rs_half = np.linspace(_R_min_sol, _R_max_sol, 1000)
        Zp = Z_boundary_sol(Rs_half)
        Rb_plot = np.concatenate([Rs_half, Rs_half[::-1]])
        Zb_plot = np.concatenate([Zp, -Zp[::-1]])

    # ── Figure layout ────────────────────────────────────────────────────────
    fig_w = 6 * ncols + 1
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 9))
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK)
    diag = ''
    if PHYSICS != 'SOLOVIEV_LINEAR' and LAST_DIAG:
        diag = f"  |  Picard: {LAST_DIAG['iterations']} it, ‖Pψ−ψ‖/‖ψ‖ = {LAST_DIAG['res_P']:.1e}"
    fig.suptitle(
        f'Grad–Shafranov Equilibrium  |  geometry={GEOMETRY}  |  physics={PHYSICS}'
        f'  |  ψ_b = {psi_b}' + diag,
        color=FG, fontsize=12, fontweight='bold', y=0.98
    )

    for ax, (title, arr, cmap, vmin, vmax, fmt_mode) in zip(axes, datasets):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE)

        arr_m  = np.where(np.isnan(arr), np.nan, arr)
        levels = np.linspace(vmin, vmax + 1e-30, 80)
        cf = ax.contourf(R_2d, Z_2d, arr_m, levels=levels, cmap=cmap, antialiased=True)

        # Flux contours (not on error panel)
        if fmt_mode != 'sci':
            n_contours = 12
            psi_levels = np.linspace(vmin + 0.05 * (vmax - vmin),
                                     vmin + 0.95 * (vmax - vmin), n_contours)
            ax.contour(R_2d, Z_2d, arr_m, levels=psi_levels,
                       colors=CONT, linewidths=0.6, alpha=0.35 if THEME == 'dark' else 0.6)

        # Boundary
        ax.plot(Rb_plot, Zb_plot, '--', color=FG, lw=1.1, alpha=0.7, label='Boundary')

        # Magnetic axis (interior min for ψ panel)
        if fmt_mode != 'sci':
            interior = [(k, ir, iz) for k, (ir, iz) in enumerate(coords)
                        if k not in bnd]
            k_ax, ir_ax, iz_ax = min(interior, key=lambda t: psi[t[0]])
            for sgn, lab in ((1, 'Mag. axis'), (-1, None)):
                ax.plot(rmesh[ir_ax], sgn * zmesh[iz_ax], '+', color='black', ms=15, mew=4.5)   # outline
                ax.plot(rmesh[ir_ax], sgn * zmesh[iz_ax], '+', color='white', ms=14, mew=2.2, label=lab)
            print(f"  Magnetic axis:  R = {rmesh[ir_ax]:.4f},  Z = {zmesh[iz_ax]:.4f}")

        cbar = plt.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=FG2)
        plt.setp(cbar.ax.get_yticklabels(), color=FG3, fontsize=8)

        if fmt_mode == 'sci':
            sfmt = ScalarFormatter(useMathText=True)
            sfmt.set_scientific(True)
            sfmt.set_powerlimits((-1, 1))
            cbar.ax.yaxis.set_major_formatter(sfmt)
            cbar.ax.yaxis.get_offset_text().set_color(FG3)
            cbar.ax.yaxis.get_offset_text().set_fontsize(9)
            cbar.set_label(r'$|\psi_\mathrm{num}-\psi_\mathrm{ana}|$',
                           color=FG3, fontsize=9)
        else:
            cbar.set_label(r'$\psi$', color=FG3, fontsize=11)

        ax.set_title(title, color=FG, fontsize=10, pad=8)
        ax.set_xlabel('R', color=FG2, fontsize=10)
        ax.set_ylabel('Z', color=FG2, fontsize=10)
        ax.tick_params(colors=FG2)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, facecolor=DARK, edgecolor=SPINE, labelcolor=FG,
                  loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, fname_override or f'gs_{GEOMETRY.lower()}_{PHYSICS.lower()}.png')
    plt.savefig(fname, dpi=160, bbox_inches='tight', facecolor=DARK)
    print(f"  Saved → {fname}")
    if SHOW:
        plt.show()
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION XI  ·  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    psi, rmesh, zmesh, inside, node_id, coords, bnd = solve()
    make_plots(psi, rmesh, zmesh, inside, coords, bnd)