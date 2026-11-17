"""Verificación del equipo de cada track del ATD con el color de la camiseta.

Un frame por segundo: para cada jugador del ATD se miran los píxeles de su torso (caja por
homografía), se descartan los parecidos al césped local, y cada píxel vota Barça o Athletic según
los colores de camiseta aprendidos sin etiquetas (data/derived/entity_colors.json, de
explore/entity_colors.py). Si en una ventana de +-5 s el color contradice con claridad al ATD, se
corrige el equipo de ese track en ese tramo.

    .venv/bin/python solution/team_check.py --workers 12
Salida: data/derived/solution/team_fix.npz  team (N, 46) int8 (-1 = sin jugador)
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (P, ROOT, WORK, N_FRAMES, FPS, L_M, W_M, halves, load_atd, load_homographies,  # noqa: E402
                    team_of_each_track, proj)

STEP = FPS                 # un frame por segundo
SEED_DE = 15
COLORS = os.path.join(ROOT, "data", "derived", "entity_colors.json")
_H = _PL = _PROTOS = None


def _init():
    global _H, _PL, _PROTOS
    _H = load_homographies()
    _PL, _ = load_atd()
    cols = json.load(open(COLORS))
    to_lab = lambda c: cv2.cvtColor(np.uint8([[np.clip(c, 0, 1) * 255]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    _PROTOS = {0: [to_lab(np.array(c)) for k in ("barca", "gk_barca") for c in cols.get(k, [])],
               1: [to_lab(np.array(c)) for k in ("athletic", "gk_athletic") for c in cols.get(k, [])]}


def votes_frame(img, f):
    h = _H[f]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    small = cv2.resize(lab, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, 11)
    out = np.zeros((46, 2), np.int32)
    pts = _PL[f]
    for s in np.flatnonzero(~np.isnan(pts[:, 0])):
        g = pts[s].astype(np.float64)
        u, v = proj(h, g)[0]
        ppm = np.hypot(*(proj(h, g + [1 / L_M, 0])[0] - (u, v)))
        x0, x1 = int(u - 0.25 * ppm), int(u + 0.25 * ppm)
        y0, y1 = int(v - 1.6 * ppm), int(v - 0.9 * ppm)
        if x0 < 0 or y0 < 0 or x1 >= img.shape[1] or y1 >= img.shape[0] or x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = lab[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        bgc = bg[min(bg.shape[0] - 1, int((y0 + y1) / 8)), min(bg.shape[1] - 1, int(u / 4))].astype(np.float32)
        roi = roi[np.linalg.norm(roi - bgc, axis=1) > 22]
        if len(roi) < 4:
            continue
        d0 = np.min([np.linalg.norm(roi - p, axis=1) for p in _PROTOS[0]], axis=0)
        d1 = np.min([np.linalg.norm(roi - p, axis=1) for p in _PROTOS[1]], axis=0)
        out[s, 0] = int(((d0 < SEED_DE) & (d0 < d1)).sum())
        out[s, 1] = int(((d1 < SEED_DE) & (d1 < d0)).sum())
    return out


def process(chunk):
    f0, f1 = chunk
    cap = cv2.VideoCapture(P["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    res = []
    for f in range(f0, f1 + 1):
        if (f - f0) % STEP:
            cap.grab()
            continue
        ok, img = cap.read()
        if not ok:
            break
        res.append((f, votes_frame(img, f)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t = time.time()
    chunks = []
    for _, s, e in halves():
        for c0 in range(s, e + 1, 2500):
            chunks.append((c0, min(e, c0 + 2499)))
    with Pool(a.workers, initializer=_init) as pool:
        rows = [r for part in pool.imap_unordered(process, chunks) for r in part]
    rows.sort(key=lambda r: r[0])
    fr = np.array([r[0] for r in rows])
    V = np.stack([r[1] for r in rows])                         # (S, 46, 2)
    print(f"votos en {len(fr)} frames muestreados ({time.time() - t:.0f}s)")

    players, _ = load_atd()
    atd_team = team_of_each_track().astype(np.int8)
    # votos acumulados en ventana de +-5 muestras (+-5 s)
    cs = np.vstack([np.zeros((1, 46, 2)), np.cumsum(V, 0)])
    S = len(fr)
    i0 = np.clip(np.arange(S) - 5, 0, S)
    i1 = np.clip(np.arange(S) + 6, 0, S)
    W_ = cs[i1] - cs[i0]                                       # (S, 46, 2)
    team_s = np.tile(atd_team, (S, 1))
    flip_to1 = (atd_team[None] == 0) & (W_[..., 1] >= 20) & (W_[..., 1] > 2 * W_[..., 0])
    flip_to0 = (atd_team[None] == 1) & (W_[..., 0] >= 20) & (W_[..., 0] > 2 * W_[..., 1])
    team_s[flip_to1] = 1
    team_s[flip_to0] = 0
    # a todos los frames: muestra más cercana
    team = np.full((N_FRAMES, 46), -1, np.int8)
    for _, f0, f1 in halves():
        idx = np.clip(np.searchsorted(fr, np.arange(f0, f1 + 1)), 0, S - 1)
        team[f0:f1 + 1] = team_s[idx]
    team[np.isnan(players[..., 0])] = -1
    np.savez_compressed(os.path.join(WORK, "team_fix.npz"), team=team)
    agree = ((W_[..., 0] + W_[..., 1]) >= 20)
    col_team = np.where(W_[..., 1] > W_[..., 0], 1, 0)
    ok = agree & (col_team == atd_team[None])
    print(f"tramos con color suficiente: {agree.sum()} | el color coincide con el ATD en "
          f"{100 * ok.sum() / max(1, agree.sum()):.1f}% | corregidos: {(flip_to1 | flip_to0).sum()} "
          f"({100 * (flip_to1 | flip_to0).sum() / max(1, agree.sum()):.2f}%)")


if __name__ == "__main__":
    main()
