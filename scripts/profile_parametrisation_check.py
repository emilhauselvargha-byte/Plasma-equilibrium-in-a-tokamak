"""Why the profile shapes are imposed on p' and FF' (not on p and F^2).

Compares plain Picard on the same boundary for three sources:
  (A) default:   shapes on the derivatives, S = R^2 p0 (1-psi_n^2)^2 + (alpha/2)(1-psi_n^2)
  (B) shapes on p and F^2 (exponents 2), then differentiated: S vanishes on the axis (hollow current)
  (C) same as (B) with exponents (1,1,1): p' nonzero on the axis, source independent of psi (linear problem)
  (D) same with exponents (1,2,1): p' nonzero on the axis, source depends on psi
and reports the fixed-point residuals ||P psi - psi||/||psi|| and ||P(P psi) - psi||/||psi|| after many iterations.
"""
from _common import *
import gs_fixedpoint as fp

gs.GEOMETRY, gs.PHYSICS, gs.R0, gs.a_sha = 'SHAFRANOV', 'CUSTOM', 0.95, 0.6
gs.kappa_sha, gs.delta_sha, gs.p0, gs.alpha = 1.7, 0.33, 1.0, 1.0

def src_deriv(r, pn):                         # (A)
    pn = np.clip(pn, 0, 1); return r**2 * gs.p0 * (1 - pn**2)**2 + 0.5 * gs.alpha * (1 - pn**2)

def src_on_p(a, b, c):                        # (B), (C): shapes on p and F^2 then differentiated
    p_shape = lambda pn: (1 - np.clip(pn, 0, 1)**a)**b
    g2_shape = lambda pn: 1 - np.clip(pn, 0, 1)**c
    return lambda r, pn: -(r**2 * gs.p0 * gs._deriv(p_shape, pn) + 0.5 * gs.alpha * gs._deriv(g2_shape, pn))

cases = [("(A) shapes on p', FF'  (default)", src_deriv),
         ("(B) shapes on p, F^2, exponents (2,2,2)", src_on_p(2, 2, 2)),
         ("(C) shapes on p, F^2, exponents (1,1,1)", src_on_p(1, 1, 1)),
         ("(D) shapes on p, F^2, exponents (1,2,1)", src_on_p(1, 2, 1))]
rows = []
for ds in [0.10, 0.05, 0.035]:
    gs.ds = ds
    for name, fn in cases:
        gs.source_custom = fn; gs.MAX_ITER = 800; gs.PICARD_EPS = 1e-10
        with quiet(): gs.solve()
        d = gs.LAST_DIAG
        rows.append(dict(case=name, ds=ds, iterations=d['iterations'], res_P=float(d['res_P']), res_P2=float(d['res_P2'])))
        print(f"ds={ds:5.3f} {name:42s} it={d['iterations']:4d}  |P psi - psi|={d['res_P']:.2e}  |P^2 psi - psi|={d['res_P2']:.2e}", flush=True)
save_json('profile_parametrisation_check.json', rows)
