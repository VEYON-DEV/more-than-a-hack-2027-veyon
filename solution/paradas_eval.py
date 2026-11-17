"""Mide corridas de paradas_tcn.py en el banco (partido afuera, evaluador oficial).

    .venv/bin/python solution/paradas_eval.py --sub bench_v3rjf --runs mdv3rjf_m1 mdv3rjf_m2 mdv3rjf_m3 --matrix
    .venv/bin/python solution/paradas_eval.py --sub bench_v3rjf --runs base15v3_m1 ... --gb-switch switch__bench_v3__bench_v3rjf.npz
  --matrix   tabla: cabeza (definición con la que se predice) x verdad (definición con la que se mide)
  --ens      además de la media de las corridas una por una, mide el promedio de sus probabilidades
"""
import argparse
import os

import numpy as np

import paradas_lib as fl

STD = (0.4, 2.0, 0.6, 1.5)          # decodificador de la entrega v14: temperatura, alpha, beta, peso del detector


def paths_for(P, S, k, tau_def, cfg, md=None, gb=None):
    out = {}
    for (m, h), p in P.items():
        s = gb[f"{m}_sw{h}"] if gb is not None else S[(m, h)][:, k]
        out[(m, h)] = fl.decode(p[:, k], s, m, tau_def, cfg, md)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="bench_v3rjf")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--ens", action="store_true")
    ap.add_argument("--head", type=float, default=1.5)
    ap.add_argument("--truth", type=float, default=1.5)
    ap.add_argument("--gb-switch", default=None)
    ap.add_argument("--cfg", type=float, nargs=4, default=list(STD))
    ap.add_argument("--save", default=None, help="guardar spellF1 por partido (npz) para comparaciones pareadas")
    a = ap.parse_args()
    gb = np.load(a.gb_switch if os.path.isabs(a.gb_switch) or os.path.exists(a.gb_switch) else os.path.join(fl.OUT, a.gb_switch)) if a.gb_switch else None
    cfg = tuple(a.cfg)
    sets = [[r] for r in a.runs] + ([a.runs] if a.ens and len(a.runs) > 1 else [])
    res = {}
    for runs in sets:
        P, S, taus = fl.load_runs(runs, a.sub)
        name = "+".join(runs) if len(runs) == 1 else f"ENS({len(runs)})"
        if a.matrix:
            for k, td in enumerate(taus):
                pt = paths_for(P, S, k, td, cfg, gb=gb if td == 1.5 else None)
                for tt in taus:
                    res.setdefault((td, tt), {})[name] = fl.per_match(pt, tt)
        else:
            k = taus.index(a.head)
            res.setdefault((a.head, a.truth), {})[name] = fl.per_match(paths_for(P, S, k, a.head, cfg, gb=gb), a.truth)
    names = [n for n in next(iter(res.values()))]
    singles = [n for n in names if not n.startswith("ENS")]
    for (td, tt), d in sorted(res.items()):
        one = np.mean([d[n] for n in singles], 0)
        line = f"cabeza {td:<4} verdad {tt:<4} | media de {len(singles)} corridas {one.mean():.4f} (por corrida {' '.join(f'{d[n].mean():.4f}' for n in singles)})"
        if any(n.startswith("ENS") for n in names):
            e = d[[n for n in names if n.startswith("ENS")][0]]
            line += f" | ensamble {e.mean():.4f} {np.round(e, 3).tolist()}"
        else:
            line += f" | por partido {np.round(one, 3).tolist()}"
        print(line)
    if a.save:
        np.savez(os.path.join(fl.OUT, a.save), **{f"{td}_{tt}_{n}": v for (td, tt), d in res.items() for n, v in d.items()})


if __name__ == "__main__":
    main()
