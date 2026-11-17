"""Comparación pareada por partido entre dos conjuntos de corridas (mismo examen, misma verdad).

    .venv/bin/python solution/paradas_cmp.py --sub bench_v3rjf --truth 1.5 \
        --a base15v3_m1 base15v3_m2 base15v3_m3 --a-head 1.5 --a-gb switch__bench_v3__bench_v3rjf.npz \
        --b mc_m1 mc_m2 mc_m3 --b-head 1.5
Imprime la media de las corridas una por una y el ensamble (probabilidades promediadas) de cada lado, y la diferencia B - A.
"""
import argparse
import os

import numpy as np

import paradas_lib as fl
from paradas_eval import paths_for, STD


def side(runs, sub, head, truth, cfg, gbf, md, ens_only=False):
    gb = np.load(gbf if os.path.exists(gbf) else os.path.join(fl.OUT, gbf)) if gbf else None
    per = []
    if not ens_only:
        for r in runs:
            P, S, taus = fl.load_runs([r], sub)
            per.append(fl.per_match(paths_for(P, S, taus.index(head), head, cfg, md, gb), truth))
    P, S, taus = fl.load_runs(runs, sub)
    ens = fl.per_match(paths_for(P, S, taus.index(head), head, cfg, md, gb), truth)
    return (np.mean(per, 0) if per else ens), ens, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="bench_v3rjf")
    ap.add_argument("--truth", type=float, default=1.5)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--a-head", type=float, default=1.5)
    ap.add_argument("--b-head", type=float, default=1.5)
    ap.add_argument("--a-gb", default=None)
    ap.add_argument("--b-gb", default=None)
    ap.add_argument("--a-md", type=float, default=None)
    ap.add_argument("--b-md", type=float, default=None)
    ap.add_argument("--a-cfg", type=float, nargs=4, default=list(STD))
    ap.add_argument("--b-cfg", type=float, nargs=4, default=list(STD))
    a = ap.parse_args()
    am, ae, ap_ = side(a.a, a.sub, a.a_head, a.truth, tuple(a.a_cfg), a.a_gb, a.a_md)
    bm, be, bp_ = side(a.b, a.sub, a.b_head, a.truth, tuple(a.b_cfg), a.b_gb, a.b_md)
    print(f"examen {a.sub}, verdad {a.truth}")
    print(f"  A cabeza {a.a_head}: media corridas {am.mean():.4f} ({' '.join(f'{p.mean():.4f}' for p in ap_)}) | ensamble {ae.mean():.4f} {np.round(ae, 3).tolist()}")
    print(f"  B cabeza {a.b_head}: media corridas {bm.mean():.4f} ({' '.join(f'{p.mean():.4f}' for p in bp_)}) | ensamble {be.mean():.4f} {np.round(be, 3).tolist()}")
    print(f"  B - A (media de corridas): {fl.paired(bm, am)}")
    print(f"  B - A (ensambles):         {fl.paired(be, ae)}")


if __name__ == "__main__":
    main()
