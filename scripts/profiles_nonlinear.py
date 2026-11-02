"""Nonlinear Picard equilibria for several boundary shapes and (p', FF') profiles.
Profiles are shaped on the derivatives:  -p'(psi_n) = p0 (1-psi_n^sp1)^sp2,  -(F^2)'(psi_n) = alpha (1-psi_n^sf1)^sf2."""
from _common import *
import matplotlib.pyplot as plt

gs.GEOMETRY, gs.PHYSICS = 'SHAFRANOV', 'PICARD_NONLINEAR'
gs.MAX_ITER, gs.ds, gs.PICARD_EPS = 500, 0.05, 1e-10
gs.R0, gs.a_sha = 0.95, 0.6

def full_field(psi, rm, zm, ins, co):
    Nr, Nz = ins.shape
    up = np.full((Nr, Nz), np.nan)
    for k, (ir, iz) in enumerate(co): up[ir, iz] = psi[k]
    Z = np.concatenate([-zm[::-1], zm[1:]])
    F = np.concatenate([up[:, ::-1], up[:, 1:]], axis=1)
    R, Zg = np.meshgrid(rm, Z, indexing='ij')
    return R, Zg, F

def solve_case(label, kappa, delta, sp1=2, sp2=2, sf1=2, sf2=1, p0=1.0, alpha=1.0):
    gs.kappa_sha, gs.delta_sha = kappa, delta
    gs.sp1, gs.sp2, gs.sf1, gs.sf2, gs.p0, gs.alpha = sp1, sp2, sf1, sf2, p0, alpha
    with quiet():
        psi, rm, zm, ins, nid, co, bnd = gs.solve()
    d = dict(gs.LAST_DIAG)
    psi_n = psi / gs.psi_b
    interior = [k for k in range(len(co)) if k not in bnd]
    kax = min(interior, key=lambda k: psi_n[k]); ir, iz = co[kax]
    mid = sorted((rm[i], k) for k, (i, j) in enumerate(co) if j == 0)
    R_in, R_out = mid[0][0], mid[-1][0]
    R_geo, hw = 0.5 * (R_in + R_out), 0.5 * (R_out - R_in)
    # sub-grid axis position: vertex of the parabola through the three midplane nodes around the minimum
    Rm = np.array([m[0] for m in mid]); Fm = np.array([psi_n[m[1]] for m in mid]); im = int(np.argmin(Fm))
    if 0 < im < len(Rm) - 1:
        c = np.polyfit(Rm[im-1:im+2], Fm[im-1:im+2], 2); R_axis = -c[1] / (2 * c[0])
    else:
        R_axis = Rm[im]
    # toroidal current density profile along the midplane  J_phi ∝ S/R   (S = R^2 p0 p_shape + alpha/2 f2_shape)
    jmid = [(R, gs.source_picard(R, psi_n[k]) / R) for R, k in mid]
    out = dict(label=label, kappa=kappa, delta=delta, sp1=sp1, sp2=sp2, sf1=sf1, sf2=sf2, p0=p0, alpha=alpha,
               N=len(co), iterations=d['iterations'], res_P=float(d['res_P']),
               R_axis=float(R_axis), Z_axis=float(zm[iz]), R_geo=float(R_geo),
               shift_over_halfwidth=float((R_axis - R_geo) / hw))
    print(f"{label:34s} N={len(co):5d} it={d['iterations']:3d} res={d['res_P']:.1e} axis R={R_axis:.3f} shift/hw={out['shift_over_halfwidth']:+.3f}")
    return out, (psi_n, rm, zm, ins, co, bnd), jmid

# -------- Study 1: shape scan at baseline profiles
shapes = [("circular k=1.0 d=0.0", 1.0, 0.0), ("ITER-like k=1.7 d=0.33", 1.7, 0.33), ("strong D k=3.0 d=0.8", 3.0, 0.8)]
rows, sols = [], []
for lab, k, d in shapes:
    o, s, _ = solve_case(lab, k, d); rows.append(o); sols.append((lab, s))

# -------- Study 2: profile scan at the ITER-like shape
profs = [("baseline", dict()),
         ("peaked p' sp=(2,4)", dict(sp2=4)),
         ("broad p',FF' sp=(4,1) sf=(4,1)", dict(sp1=4, sp2=1, sf1=4, sf2=1)),
         ("current-driven p0=0.2 a=2", dict(p0=0.2, alpha=2.0)),
         ("pressure-driven p0=3 a=0.3", dict(p0=3.0, alpha=0.3))]
psols, jprof = [], []
for lab, kw in profs:
    o, s, j = solve_case(lab, 1.7, 0.33, **kw); rows.append(o); psols.append((lab, s)); jprof.append((lab, j))
save_json('nonlinear_cases.json', rows)

# -------- Figures
def gallery(items, fname, title):
    n = len(items); fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 5.2), constrained_layout=True)
    for ax, (lab, (psi, rm, zm, ins, co, bnd)) in zip(np.atleast_1d(axes), items):
        R, Z, F = full_field(psi, rm, zm, ins, co)
        cf = ax.contourf(R, Z, F, levels=np.linspace(0, 1, 41), cmap='plasma')
        ax.contour(R, Z, F, levels=np.linspace(0.05, 0.95, 10), colors='w', linewidths=.5, alpha=.5)
        k = min([k for k in range(len(co)) if k not in bnd], key=lambda k: psi[k]); ir, iz = co[k]
        ax.plot(rm[ir], zm[iz], 'w+', ms=10, mew=2); ax.plot(rm[ir], -zm[iz], 'w+', ms=10, mew=2)
        ax.set_aspect('equal'); ax.set_title(lab, fontsize=8); ax.set_xlabel('R')
    np.atleast_1d(axes)[0].set_ylabel('Z'); fig.colorbar(cf, ax=axes, shrink=.6, label=r'$\psi_n$'); fig.suptitle(title, fontsize=10)
    fig.savefig(os.path.join(FIG, fname), dpi=170); plt.close(fig)
gallery(sols, 'nonlinear_shapes.png', r'Picard equilibria, baseline profiles, varying boundary shape')
gallery(psols, 'nonlinear_profiles.png', r"Picard equilibria, ITER-like boundary, varying $p'(\psi)$, $FF'(\psi)$ profiles")

fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
x = np.linspace(0, 1, 200)
for lab, kw in profs:
    a, b = kw.get('sp1', 2), kw.get('sp2', 2)
    axs[0].plot(x, (1 - x**a)**b, label=lab)
axs[0].set_xlabel(r'$\psi_n$'); axs[0].set_ylabel(r"$-p'/p_0$"); axs[0].set_title(r"$p'$ shape functions"); axs[0].legend(fontsize=6)
for lab, (psi, rm, zm, ins, co, bnd) in psols:
    m = sorted((rm[i], psi[k]) for k, (i, j) in enumerate(co) if j == 0)
    axs[1].plot([a for a, _ in m], [b for _, b in m], label=lab)
axs[1].set_xlabel('R (midplane)'); axs[1].set_ylabel(r'$\psi_n$'); axs[1].set_title('midplane flux profile')
for lab, j in jprof:
    jj = np.array([v for _, v in j]); axs[2].plot([r for r, _ in j], jj / jj.max(), label=lab)
axs[2].set_xlabel('R (midplane)'); axs[2].set_ylabel(r'$J_\varphi/J_{\varphi,\max}$'); axs[2].set_title('toroidal current density (midplane)')
axs[3].bar(range(5), [r['shift_over_halfwidth'] for r in rows[3:]], color='#2a6fdb'); axs[3].set_xticks(range(5))
axs[3].set_xticklabels(['base', "peaked p'", 'broad', 'I-driven', 'p-driven'], fontsize=7, rotation=20)
axs[3].set_ylabel('axis shift / half-width'); axs[3].set_title('Shafranov-type shift')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'nonlinear_profile_diagnostics.png'), dpi=170); plt.close(fig)
