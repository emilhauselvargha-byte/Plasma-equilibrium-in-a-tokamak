"""Shared helpers for the study scripts (path setup, quiet solves, mesh plotting)."""
import os, sys, io, contextlib, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import gs_solver as gs

FIG = os.path.join(ROOT, 'figures')
RES = os.path.join(ROOT, 'results')
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
gs.OUTPUT_DIR = FIG

@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def build(verbose=False):
    """Boundary sampling -> grid -> sparse matrix. Returns a dict of all objects."""
    cm = contextlib.nullcontext() if verbose else quiet()
    with cm:
        rs, zs = gs.sample_outer_boundary()
        rm, zm, ins, _, _ = gs.build_grid(rs, zs)
        nid, co = gs.build_index_map(ins)
        A, bnd = gs.assemble(rm, zm, ins, nid, co)
    return dict(rs=rs, zs=zs, rm=rm, zm=zm, ins=ins, nid=nid, co=co, A=A, bnd=bnd)

def save_json(name, obj):
    with open(os.path.join(RES, name), 'w') as f:
        json.dump(obj, f, indent=2)
