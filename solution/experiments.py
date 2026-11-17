"""Experimentos de mejora sobre el banco de IDSSE (dejando un partido afuera cada vez).

  ref        modelo actual (gradient boosting + Markov)
  mindur     + duración mínima de posesión (los tramos < MIN_DUR se unen al anterior)
  long       + pistas en ventanas largas (+-5 s, +-10 s) del primer modelo
  seeds      + entrenar con varias semillas de ensuciado (más variedad de errores)
  stack      + segundo modelo sobre las probabilidades del primero en +-10 s

    .venv/bin/python solution/experiments.py --exp mindur long stack
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, BENCH_SUB, FPS  # noqa: E402
from features import window  # noqa: E402
import features as featmod  # noqa: E402
from bench import (match_features, make_model, transitions_from, hmm_decode, score, controlled,  # noqa: E402
                   TAU_CTRL)

MIN_DUR = TAU_CTRL


def min_duration(path, min_f):
    """Tramos de H/A más cortos que min_f: siguen con el equipo anterior (o pasan al siguiente
    si vienen justo después de una parada). Misma regla que la definición de las etiquetas."""
    return controlled(path, min_f)


LONG_SRC = ("ddiff", "ctrlH_c", "ctrlA_c", "bspeed", "pspeed", "known", "out_dist", "bx_home", "cx_diff",
            "vcx_diff", "behind_diff", "press_diff", "area_diff", "close_diff", "tarrive_diff", "bvx_home",
            "ball_still", "kick_from_rest", "n_9m_H", "n_9m_A")


def add_long(X, names):
    cols = []
    for nm in [n for n in LONG_SRC if n in names]:   # las de pelota parada solo existen con SETPIECE=1
        x = X[:, names.index(nm)].astype(np.float64)
        for a, b in ((-125, -1), (1, 125), (-250, -126), (126, 250)):
            cols.append(window(x, a, b))
    return np.hstack([X, np.stack(cols, 1).astype(np.float32)])


def stack_feats(P):
    cols = [P]
    for k in range(3):
        for a, b in ((-12, 12), (-50, -1), (1, 50), (-125, -51), (51, 125), (-250, 250)):
            cols.append(window(P[:, k].astype(np.float64), a, b)[:, None])
    return np.hstack(cols).astype(np.float32)


def run(exps, taus=(0.1, 0.2, 0.35)):
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz")))
    mids = [os.path.basename(p)[:-4] for p in paths]
    seeds = (1, 2, 3) if "seeds" in exps else (1,)
    data = {s: {m: match_features(p, seed=s) for m, p in zip(mids, paths)} for s in seeds}
    names = list(featmod.NAMES)
    use_long = "long" in exps
    if use_long:
        for s in seeds:
            for m in mids:
                X2, y2, r2 = data[s][m]
                data[s][m] = ([add_long(x, names) for x in X2], y2, r2)
    print(f"datos listos ({time.time() - t0:.0f}s); semillas {seeds}; pistas {data[1][mids[0]][0][0].shape[1]}")
    results = {}
    for mid in mids:
        train = [m for m in mids if m != mid]
        Xtr = np.vstack([x[::3] for s in seeds for m in train for x in data[s][m][0]])
        ytr = np.concatenate([y[::3] for s in seeds for m in train for y in data[s][m][1]])
        model = make_model().fit(Xtr, ytr)
        prior = np.bincount(ytr, minlength=3) / len(ytr)
        T = transitions_from([y for m in train for y in data[1][m][1]])
        Xte, yte = data[1][mid][0], data[1][mid][1]
        P = [model.predict_proba(x) for x in Xte]
        if "stack" in exps:
            # probabilidades fuera de muestra para los partidos de entrenamiento (CV interna)
            Ztr, ytr2 = [], []
            groups = [train[i::3] for i in range(3)]          # CV interna de 3 grupos (más rápida)
            for grp in groups:
                tr2 = [m for m in train if m not in grp]
                Xi = np.vstack([x[::4] for m in tr2 for x in data[1][m][0]])
                yi = np.concatenate([y[::4] for m in tr2 for y in data[1][m][1]])
                mi = make_model().fit(Xi, yi)
                for inner in grp:
                    for x, y in zip(data[1][inner][0], data[1][inner][1]):
                        Ztr.append(stack_feats(mi.predict_proba(x))[::3])
                        ytr2.append(y[::3])
            m2 = make_model().fit(np.vstack(Ztr), np.concatenate(ytr2))
            P = [m2.predict_proba(stack_feats(p)) for p in P]
        for tau in taus:
            paths_ = [hmm_decode(p, prior, T, tau) for p in P]
            key = f"tau={tau}"
            results.setdefault(key, []).append(score(yte, paths_))
            if "mindur" in exps:
                paths2 = [min_duration(pp, int(MIN_DUR * FPS)) for pp in paths_]
                results.setdefault(key + "+mindur", []).append(score(yte, paths2))
        print(f"  {mid}: " + "  ".join(f"{k}={v[-1][0]:.3f}" for k, v in results.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nexperimento: {' + '.join(exps) or 'ref'}")
    out = {}
    for k, v in results.items():
        a = np.array(v)
        out[k] = float(a[:, 0].mean())
        print(f"  {k:18} spellF1 {a[:, 0].mean():.3f}  macroF1 {a[:, 1].mean():.3f}  posesiones {a[:, 2].mean():.0f}/{a[:, 3].mean():.0f}"
              f"   ({' '.join(f'{x:.2f}' for x in a[:, 0])})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=[])
    a = ap.parse_args()
    res = run(a.exp)
    tag = "_".join(a.exp) or "ref"
    json.dump(res, open(os.path.join(WORK, BENCH_SUB, f"exp_{tag}.json"), "w"), indent=1)
