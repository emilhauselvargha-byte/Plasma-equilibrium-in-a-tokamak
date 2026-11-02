"""Validation of the nonlinear (Picard) code path.
 (i)  Soloviev through the Picard path: PHYSICS='CUSTOM' with the exact Soloviev source (psi-independent).
      The iteration must converge immediately and reproduce the analytic solution (same boundary/stencils as the linear benchmark).
 (ii) Grid convergence of a genuinely nonlinear case: the amplitude functional C = 1 - u_min (u = unnormalised solve at the
      fixed point) on the ITER-like boundary, with a curve fit  C(ds) = C_inf + c ds^p.
"""
from _common import *
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------------- (i)
gs.GEOMETRY, gs.PHYSICS, gs.PICARD_EPS, gs.MAX_ITER = 'SOLOVIEV', 'CUSTOM', 1e-12, 50
gs.source_custom = lambda r, pn: gs.source_soloviev(r)
rowsA = []
for ds in [0.14, 0.10, 0.07, 0.05, 0.035, 0.025]:
    gs.ds = ds
    with quiet(): psi, rm, zm, ins, nid, co, bnd = gs.solve()
    ana = np.array([gs.psi_analytic_soloviev(rm[ir], zm[iz]) for ir, iz in co]); err = np.abs(psi - ana)
    rowsA.append(dict(ds=ds, N=len(co), iterations=gs.LAST_DIAG['iterations'], res_P=float(gs.LAST_DIAG['res_P']), max_err=float(err.max()), mean_err=float(err.mean())))
    print(rowsA[-1])
pm = np.polyfit(np.log([r['ds'] for r in rowsA]), np.log([r['max_err'] for r in rowsA]), 1)[0]
print("Soloviev via Picard path: observed order (max) = %.2f" % pm)

# ---------------- (ii)
gs.GEOMETRY, gs.PHYSICS, gs.PICARD_EPS, gs.MAX_ITER = 'SHAFRANOV', 'PICARD_NONLINEAR', 1e-12, 500
gs.R0, gs.a_sha, gs.kappa_sha, gs.delta_sha = 0.95, 0.6, 1.7, 0.33
gs.sp1, gs.sp2, gs.sf1, gs.sf2, gs.p0, gs.alpha = 2, 2, 2, 1, 1.0, 1.0
rowsB = []
for ds in [0.14, 0.10, 0.07, 0.05, 0.035, 0.025, 0.018]:
    gs.ds = ds
    with quiet(): psi, rm, zm, ins, nid, co, bnd = gs.solve()
    x = psi / gs.psi_b; N = len(co)
    with quiet():
        rs, zs = gs.sample_outer_boundary(); rm2, zm2, ins2, _, _ = gs.build_grid(rs, zs)
        nid2, co2 = gs.build_index_map(ins2); A, bnd2 = gs.assemble(rm2, zm2, ins2, nid2, co2)
    b = np.ones(N)
    for k, (ir, iz) in enumerate(co):
        if k not in bnd: b[k] = gs.source_picard(rm[ir], np.clip(x[k], 0, 1))
    u = spsolve(A, b); C = 1.0 - u.min()
    rowsB.append(dict(ds=ds, N=N, iterations=gs.LAST_DIAG['iterations'], res_P=float(gs.LAST_DIAG['res_P']), C=float(C)))
    print(rowsB[-1], flush=True)
ds_ = np.array([r['ds'] for r in rowsB]); C_ = np.array([r['C'] for r in rowsB])
f = lambda d, Cinf, c, p: Cinf + c * d**p
popt, _ = curve_fit(f, ds_, C_, p0=[C_[-1], 1.0, 2.0], maxfev=20000)
print("fit: C_inf=%.6f  c=%.3g  p=%.2f" % tuple(popt))
save_json('nonlinear_validation.json', dict(soloviev_via_picard=rowsA, soloviev_order_max=float(pm), amplitude_convergence=rowsB,
                                           fit=dict(C_inf=float(popt[0]), c=float(popt[1]), p=float(popt[2]))))
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.loglog(ds_, np.abs(C_ - popt[0]), 'o-', label=r'$|C(\Delta s)-C_\infty|$ (fit $p=%.2f$)' % popt[2])
ax.loglog(ds_, np.abs(C_[0] - popt[0]) * (ds_ / ds_[0])**2, 'k:', label=r'$\propto\Delta s^2$')
ax.set_xlabel(r'$\Delta s$'); ax.set_ylabel('error of amplitude functional $C=1-u_{\\min}$'); ax.grid(True, which='both', alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'nonlinear_convergence.png'), dpi=200)
