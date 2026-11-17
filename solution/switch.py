"""Detector de cambios + Markov con transiciones moduladas frame a frame (banco de IDSSE).

El análisis de errores mostró que lo que más falla son los cambios de posesión con la pelota viva
(equipo -> equipo: se pierde ~la mitad) y los saques (muerto -> vivo). Aquí:

  1. pistas de "qué cambia": para señales clave, (media del segundo siguiente) - (media del anterior)
  2. dos detectores binarios (gradient boosting), dejando un partido afuera:
       sw: "en +-3 frames hay un cambio H<->A en la verdad"
       rs: "en +-3 frames se reanuda el juego (D -> H/A)"
  3. Viterbi con transiciones moduladas: el cambio H<->A en el frame t se multiplica por
     (p_sw[t] / media)^g_sw, y la reanudación por (p_rs[t] / media)^g_rs.

    .venv/bin/python solution/switch.py --tag v6
Guarda data/derived/solution/bench/switch_<tag>.npz (probabilidades de los detectores) y
switch_<tag>.json (mejor configuración).
"""

import argparse
import glob
import itertools
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
from bench import match_features, adjust_T, score, njit, controlled  # noqa: E402
from experiments import add_long, min_duration  # noqa: E402

CHANGE_SRC = ("ddiff", "ctrlH_c", "ctrlA_c", "close_diff", "tarrive_diff", "bvx_home", "bspeed",
              "pspeed", "known", "press_diff", "behind_diff", "vcx_diff", "chasers_diff", "n2mH", "n2mA")


def change_feats(X, names):
    cols = []
    for nm in CHANGE_SRC:
        x = X[:, names.index(nm)].astype(np.float64)
        for w in (12, 25, 50):
            cols.append(window(x, 1, w) - window(x, -w, -1))
    return np.hstack([X, np.stack(cols, 1).astype(np.float32)])


def near_transition(y, kind, tol=3):
    """1 donde hay (en +-tol frames) una transición del tipo pedido en la verdad."""
    t = np.flatnonzero(y[1:] != y[:-1]) + 1
    a, b = y[t - 1], y[t]
    if kind == "sw":
        sel = (a != 2) & (b != 2)
    elif kind == "rs":
        sel = (a == 2) & (b != 2)
    else:
        sel = (a != 2) & (b == 2)
    out = np.zeros(len(y), np.int8)
    for tt in t[sel]:
        out[max(0, tt - tol):tt + tol + 1] = 1
    return out


def make_det():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=48,
                                          min_samples_leaf=100, l2_regularization=1.0, random_state=0)


@njit(cache=True)
def _viterbi_tv(logp, logT, lsw, lrs):
    n = logp.shape[0]
    score_ = logp[0].copy()
    back = np.zeros((n, 3), np.int8)
    new = np.empty(3)
    M = np.empty((3, 3))
    for t in range(1, n):
        for i in range(3):
            for j in range(3):
                M[i, j] = logT[i, j]
        M[0, 1] += lsw[t]
        M[1, 0] += lsw[t]
        M[2, 0] += lrs[t]
        M[2, 1] += lrs[t]
        for j in range(3):
            best, arg = -1e300, 0
            for i in range(3):
                v = score_[i] + M[i, j]
                if v > best:
                    best, arg = v, i
            back[t, j] = arg
            new[j] = best + logp[t, j]
        score_[:] = new
    path = np.zeros(n, np.int8)
    path[n - 1] = np.argmax(score_)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_tv(P, prior, T, tau, alpha, beta, psw, prs, g_sw, g_rs):
    loglik = tau * (np.log(np.clip(P, 1e-6, 1)) - np.log(prior))
    lT = np.log(adjust_T(T, alpha, beta))
    lsw = g_sw * (np.log(np.clip(psw, 1e-5, 1)) - np.log(max(psw.mean(), 1e-5)))
    lrs = g_rs * (np.log(np.clip(prs, 1e-5, 1)) - np.log(max(prs.mean(), 1e-5)))
    return _viterbi_tv(np.ascontiguousarray(loglik), np.ascontiguousarray(lT),
                       np.clip(lsw, -12, 12).astype(np.float64), np.clip(lrs, -12, 12).astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tau-train", type=float, default=3.0, help="control mínimo (s) de las etiquetas de entrenamiento")
    a = ap.parse_args()
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz")))
    mids = [os.path.basename(p)[:-4] for p in paths]
    data = {}
    for s in a.seeds:
        for m, p in zip(mids, paths):
            X2, _, r2 = match_features(p, seed=s)
            names = list(featmod.NAMES)
            X2 = [add_long(x, names) for x in X2]
            y2 = [controlled(r, int(a.tau_train * FPS)) for r in r2]
            data[(s, m)] = ([change_feats(x, names) for x in X2], y2)
    print(f"pistas de cambio listas ({time.time() - t0:.0f}s), {data[(a.seeds[0], mids[0])][0][0].shape[1]} columnas", flush=True)
    det = {}
    rng = np.random.default_rng(0)
    for mid in mids:
        tr = [m for m in mids if m != mid]
        det[mid] = {}
        for kind in ("sw", "rs"):
            Xs, ys = [], []
            for s in a.seeds:
                for m in tr:
                    for x, y in zip(*data[(s, m)]):
                        t = near_transition(y, kind)
                        keep = (t == 1) | (rng.random(len(t)) < 0.25)
                        Xs.append(x[keep])
                        ys.append(t[keep])
            model = make_det().fit(np.vstack(Xs), np.concatenate(ys))
            det[mid][kind] = [model.predict_proba(x)[:, 1] for x in data[(a.seeds[0], mid)][0]]
        print(f"  detectores {mid} listos ({time.time() - t0:.0f}s)", flush=True)
    save = {f"{m}_{k}{h}": det[m][k][h] for m in mids for k in ("sw", "rs") for h in range(2)}
    np.savez_compressed(os.path.join(WORK, BENCH_SUB, f"switch_{a.tag}.npz"), **save)

    z = np.load(os.path.join(WORK, BENCH_SUB, f"probas_{a.tag}.npz"))
    grid = list(itertools.product((0.2, 0.3, 0.5), (1.0, 2.0), (0.3, 1.0), (0.0, 1.0, 1.5, 2.0), (0.0, 0.5, 1.0),
                                  (0, 1, 1.5, 2, 3)))
    res = []
    for tau, al, be, gsw, grs, md in grid:
        sc = []
        for m in mids:
            paths_ = []
            for h in range(2):
                p = decode_tv(z[f"{m}_P{h}"], z[f"{m}_prior"], z[f"{m}_T"], tau, al, be,
                              det[m]["sw"][h], det[m]["rs"][h], gsw, grs)
                if md:
                    p = min_duration(p, int(md * FPS))
                paths_.append(p)
            sc.append(score([z[f"{m}_y0"], z[f"{m}_y1"]], paths_))
        sc = np.array(sc)
        res.append((float(sc[:, 0].mean()), float(sc[:, 1].mean()), tau, al, be, gsw, grs, md,
                    float(sc[:, 2].mean()), float(sc[:, 3].mean()), sc[:, 0].round(3).tolist()))
    res.sort(key=lambda r: -r[0])
    print(f"\nmejores ({time.time() - t0:.0f}s):  spellF1 macroF1  tau alpha beta g_sw g_rs mindur  posesiones")
    for r in res[:8]:
        print(f"  {r[0]:.3f}  {r[1]:.3f}   {r[2]} {r[3]} {r[4]} {r[5]} {r[6]} {r[7]}   {r[8]:.0f}/{r[9]:.0f}  {r[10]}")
    base = [r for r in res if r[5] == 0 and r[6] == 0]
    print(f"  (sin detectores, mejor: {base[0][0]:.3f})")
    b = res[0]
    json.dump({"tau": b[2], "alpha": b[3], "beta": b[4], "g_sw": b[5], "g_rs": b[6], "mindur": b[7],
               "spellF1": b[0], "macroF1": b[1]}, open(os.path.join(WORK, BENCH_SUB, f"switch_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
