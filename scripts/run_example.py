"""Minimal usage example: one nonlinear equilibrium (ITER-like boundary) and one Solov'ev benchmark."""
from _common import *

# 1) Linear Solov'ev benchmark
gs.GEOMETRY, gs.PHYSICS, gs.ds = 'SOLOVIEV', 'SOLOVIEV_LINEAR', 0.03
psi, rm, zm, ins, nid, co, bnd = gs.solve()
gs.make_plots(psi, rm, zm, ins, co, bnd, fname_override='example_soloviev.png')

# 2) Nonlinear plain Picard on a parametric D-shape
gs.GEOMETRY, gs.PHYSICS, gs.ds = 'SHAFRANOV', 'PICARD_NONLINEAR', 0.05
gs.kappa_sha, gs.delta_sha, gs.a_sha = 1.7, 0.33, 0.6
psi, rm, zm, ins, nid, co, bnd = gs.solve()
gs.make_plots(psi, rm, zm, ins, co, bnd, fname_override='example_nonlinear.png')
