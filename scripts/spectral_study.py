"""Spectrum of the plain Picard map P at its fixed point (PHYSICS='PICARD_NONLINEAR').

Near a fixed point psi* the error evolves as e_{k+1} ~ L e_k with L = dP/dpsi (dense Jacobian by central differences),
so plain Picard converges iff rho(L) < 1, in about log(tol)/log(rho(L)) iterations.
"""
from _common import *
import gs_fixedpoint as fp
import matplotlib.pyplot as plt

gs.GEOMETRY, gs.PHYSICS = 'SHAFRANOV', 'PICARD_NONLINEAR'
gs.R0, gs.a_sha = 0.95, 0.6
BASE = dict(sp1=2, sp2=2, sf1=2, sf2=1)

def setup(ds, kappa, delta, p0=1.0, alpha=1.0, **kw):
    for k, v in {**BASE, **kw}.items(): setattr(gs, k, v)
    gs.ds, gs.kappa_sha, gs.delta_sha, gs.p0, gs.alpha = ds, kappa, delta, p0, alpha
    b = build(); P = fp.make_map(b['A'], b['bnd'], b['rm'], b['co'])
    x0 = np.zeros(len(b['co'])); x0[list(b['bnd'])] = 1.0
    return b, P, x0

def iterate(P, x0, nmax=400, tol=1e-11):
    x, hist = x0.copy(), []
    for it in range(nmax):
        xn = P(x); r = fp.rel(xn, x); hist.append(r); x = xn
        if r < tol: break
    return x, np.array(hist)

# ---------------- main case (N = 961)
b, P, x0 = setup(0.05, 1.7, 0.33)
x, hist = iterate(P, x0); N = len(x)
lam = fp.sorted_eigs(fp.dense_jacobian(P, x)); rho = float(abs(lam[0]))
main = dict(N=N, iterations=len(hist), res_P=fp.rel(P(x), x), rho_L=rho,
            min_re_lambda=float(lam.real.min()), max_re_lambda=float(lam.real.max()), max_abs_imag=float(np.abs(lam.imag).max()),
            predicted_iterations=float(np.log(1e-11) / np.log(rho)))
print("main case:", main)

# ---------------- several configurations (coarser grids)
configs = [("ITER-like, baseline", dict(ds=0.07, kappa=1.7, delta=0.33)),
           ("circular", dict(ds=0.07, kappa=1.0, delta=0.0)),
           ("strong D (3.0, 0.8)", dict(ds=0.10, kappa=3.0, delta=0.8)),
           ("pressure-driven p0/a=10", dict(ds=0.07, kappa=1.7, delta=0.33, p0=3.0, alpha=0.3)),
           ("current-driven p0/a=0.1", dict(ds=0.07, kappa=1.7, delta=0.33, p0=0.2, alpha=2.0)),
           ("broad profiles", dict(ds=0.07, kappa=1.7, delta=0.33, sp1=4, sp2=1, sf1=4, sf2=1)),
           ("peaked p'", dict(ds=0.07, kappa=1.7, delta=0.33, sp2=4))]
table = []
for name, cfg in configs:
    bb, PP, xx0 = setup(**cfg)
    xx, h = iterate(PP, xx0)
    lm = fp.sorted_eigs(fp.dense_jacobian(PP, xx))
    row = dict(case=name, N=len(bb['co']), iterations=len(h), res_P=fp.rel(PP(xx), xx), rho_L=float(abs(lm[0])),
               min_re=float(lm.real.min()), max_abs_imag=float(np.abs(lm.imag).max()))
    table.append(row); print({k: (round(v, 4) if isinstance(v, float) and v > 1e-3 else v) for k, v in row.items()}, flush=True)
save_json('spectral_study.json', dict(main=main, table=table))

# ---------------- figure
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
a1.semilogy(np.arange(1, len(hist) + 1), hist, c='#1f77b4', label='measured')
k = np.arange(1, len(hist) + 1); a1.semilogy(k, hist[0] * rho**(k - 1), 'k--', lw=1, label=r'$\propto\rho(L)^k$, $\rho(L)=%.3f$' % rho)
a1.set_xlabel('Picard iteration $k$'); a1.set_ylabel(r"$\|P\psi^k-\psi^k\|/\|\psi^k\|$"); a1.grid(True, which='both', alpha=.3); a1.legend()
a1.set_title('Convergence of plain Picard')
th = np.linspace(0, 2 * np.pi, 400)
a2.fill(np.cos(th), np.sin(th), color='#2ca02c', alpha=.15, label=r'convergent: $|\lambda|<1$')
a2.plot(lam.real, lam.imag, 'o', c='#d62728', ms=3, label=r'eigenvalues of $L$ at $\psi^*$')
a2.axhline(0, c='k', lw=.4); a2.axvline(0, c='k', lw=.4); a2.set_aspect('equal'); a2.set_xlim(-1.2, 1.2); a2.set_ylim(-1.2, 1.2)
a2.set_xlabel(r'Re $\lambda$'); a2.set_ylabel(r'Im $\lambda$'); a2.legend(fontsize=8, loc='lower left'); a2.set_title(r'Jacobian spectrum ($N=%d$)' % N)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'picard_spectral_study.png'), dpi=180); plt.close(fig)
