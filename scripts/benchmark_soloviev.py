"""Benchmark of the linear Soloviev solve against the analytic solution + grid-convergence study."""
import time
from _common import *
import matplotlib.pyplot as plt

gs.GEOMETRY, gs.PHYSICS = 'SOLOVIEV', 'SOLOVIEV_LINEAR'

def run(ds):
    gs.ds = ds
    t0 = time.time()
    with quiet():
        psi, rm, zm, ins, nid, co, bnd = gs.solve()
    dt = time.time() - t0
    ana = np.array([gs.psi_analytic_soloviev(rm[ir], zm[iz]) for ir, iz in co])
    err = np.abs(psi - ana)
    return dict(ds=ds, N=len(co), nr=len(rm), nz=len(zm), max_err=err.max(), mean_err=err.mean(),
                rel_max=err.max() / ana.max(), time=dt), (psi, rm, zm, ins, co, bnd)

# --- headline figure (numerical | analytic | error)
res, sol = run(0.02)
print("headline:", res)
with quiet():
    gs.make_plots(*[sol[0], sol[1], sol[2], sol[3], sol[4], sol[5]], fname_override='soloviev_benchmark.png')

# --- convergence study
rows = []
for ds in [0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012]:
    r, _ = run(ds); rows.append(r)
    print(f"ds={ds:5.3f}  N={r['N']:6d}  max_err={r['max_err']:.3e}  mean_err={r['mean_err']:.3e}  t={r['time']:.2f}s")
save_json('soloviev_convergence.json', rows)

ds = np.array([r['ds'] for r in rows]); em = np.array([r['max_err'] for r in rows]); ea = np.array([r['mean_err'] for r in rows])
p_max = np.polyfit(np.log(ds), np.log(em), 1)[0]; p_mean = np.polyfit(np.log(ds), np.log(ea), 1)[0]
print(f"fitted order: max-norm {p_max:.2f}, mean {p_mean:.2f}")
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.loglog(ds, em, 'o-', label=f'max error (slope {p_max:.2f})')
ax.loglog(ds, ea, 's-', label=f'mean error (slope {p_mean:.2f})')
ax.loglog(ds, em[0] * (ds / ds[0])**2, 'k:', label=r'$\propto \Delta s^2$')
ax.set_xlabel(r'boundary arc-length step $\Delta s$'); ax.set_ylabel(r'$|\psi_{\rm num}-\psi_{\rm ana}|$')
ax.grid(True, which='both', alpha=.3); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(FIG, 'soloviev_convergence.png'), dpi=200)

# --- mesh pictures (non-homogeneous meshgrid) for both geometries
def mesh_fig(geom, name):
    gs.GEOMETRY = geom; gs.ds = 0.06
    b = build()
    Rr, Zz = [], []
    for ir, iz in b['co']:
        Rr.append(b['rm'][ir]); Zz.append(b['zm'][iz])
    Rr, Zz = np.array(Rr), np.array(Zz)
    isb = np.zeros(len(Rr), bool); isb[list(b['bnd'])] = True
    fig, ax = plt.subplots(figsize=(4.6, 6))
    for sgn in (1, -1):
        ax.plot(Rr[~isb], sgn * Zz[~isb], '.', ms=2.5, c='#2a6fdb')
        ax.plot(Rr[isb], sgn * Zz[isb], '.', ms=4, c='#d62728')
    ax.set_aspect('equal'); ax.set_xlabel('R'); ax.set_ylabel('Z'); ax.set_title(f'{geom.capitalize()} boundary: {len(Rr)} nodes (upper half, mirrored)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=200)
mesh_fig('SOLOVIEV', 'mesh_soloviev.png'); mesh_fig('SHAFRANOV', 'mesh_shafranov.png')
