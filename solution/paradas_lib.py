"""Utilidades para medir en el banco de IDSSE (partido afuera) las redes multi-definición de paradas_tcn.py
con el evaluador oficial: verdad con distintas definiciones de control, transiciones y decodificación por definición.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as pm  # noqa: E402
from common import WORK, FPS  # noqa: E402
from bench import controlled, transitions_from, swap_labels  # noqa: E402
from switch import decode_tv  # noqa: E402

OUT = os.path.join(WORK, "paradas")
MIDS = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(WORK, "idsse", "*.npz")))
_raw = {}


def raw_labels(m):
    if m not in _raw:
        z = np.load(os.path.join(WORK, "bench_v3", f"{m}_s1.npz"))
        _raw[m] = [z["raw0"], z["raw1"]]
    return _raw[m]


_truth = {}


_ctrl = None


def truth(m, tau):
    """tau > 0: control mínimo de tau s; 0: etiquetas crudas (por toques); < 0: control por contacto (paradas_ctrl.py)."""
    global _ctrl
    k = (m, tau)
    if k not in _truth:
        if tau < 0:
            if _ctrl is None:
                _ctrl = np.load(os.path.join(OUT, "ctrl_labels.npz"))
            suf = "" if tau == -1 else str(int(-tau))             # -1 base, -2 estricta, -3 laxa
            _truth[k] = [_ctrl[f"{m}_c{suf}0"], _ctrl[f"{m}_c{suf}1"]]
        else:
            _truth[k] = [controlled(r, int(round(tau * FPS))) if tau > 0 else r.copy() for r in raw_labels(m)]
    return _truth[k]


_pt = {}


def prior_T(mid, tau):
    """Frecuencias y transiciones (simétricas local/visitante) medidas en los otros 6 partidos."""
    k = (mid, tau)
    if k not in _pt:
        ys = [y for m in MIDS if m != mid for y in truth(m, tau)]
        ys = ys + [swap_labels(y) for y in ys]
        prior = np.bincount(np.concatenate([y[::3] for y in ys]), minlength=3).astype(float)
        _pt[k] = (prior / prior.sum(), transitions_from(ys))
    return _pt[k]


def fit(a, n):
    a = a[:n]
    if len(a) < n:
        a = np.concatenate([a, np.repeat(a[-1:], n - len(a), 0)])
    return a


def load_runs(runs, sub):
    """Promedia las probabilidades de varias corridas (misma lista de definiciones). -> P[(m,h)] (n,K,3), S[(m,h)] (n,K), taus"""
    zs = [np.load(os.path.join(OUT, f"{r}__{sub}.npz")) for r in runs]
    taus = [float(t) for t in zs[0]["taus"]]
    P, S = {}, {}
    for m in MIDS:
        for h in (0, 1):
            n = len(raw_labels(m)[h])
            P[(m, h)] = sum(fit(z[f"{m}_P{h}"].astype(np.float32), n) for z in zs) / len(zs)
            S[(m, h)] = sum(fit(z[f"{m}_S{h}"].astype(np.float32), n) for z in zs) / len(zs)
    return P, S, taus


def decode(P, S, mid, tau_def, cfg, md=None):
    """P (n,3), S (n,) de una definición. cfg = (temperatura, alpha, beta, peso del detector)."""
    prior, T = prior_T(mid, tau_def)
    tmp, alpha, beta, g = cfg
    path = decode_tv(np.asarray(P, np.float64), prior, T, tmp, alpha, beta, np.asarray(S, np.float64),
                     np.ones(len(P)), g, 0.0)
    md = max(tau_def, 0.0) if md is None else md
    return controlled(path, int(round(md * FPS))) if md > 0 else path


def score_match(ys, ps):
    r = pm.evaluate(np.r_[ys[0], 3, ys[1]].astype(np.int8), np.r_[ps[0], 3, ps[1]].astype(np.int8), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def per_match(paths, tau_truth):
    """paths[(m,h)] -> lista de spellF1 por partido contra la verdad con control mínimo tau_truth."""
    return np.array([score_match(truth(m, tau_truth), [paths[(m, 0)], paths[(m, 1)]])[0] for m in MIDS])


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return f"dif media {d.mean():+.4f}, gana en {int((d > 0).sum())}/7, por partido {np.round(d, 3).tolist()}"
