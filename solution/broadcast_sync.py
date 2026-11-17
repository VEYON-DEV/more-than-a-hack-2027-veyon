"""Sincroniza el tracking del broadcast con el táctico, sin video, y mapea su pelota al reloj táctico.

Los dos ATD están en coordenadas del campo: en un mismo instante real los jugadores están en el
mismo lugar en ambos. Para un frame b del broadcast y un desfase d, el costo es la distancia media
de cada jugador del broadcast al jugador más cercano DE SU MISMO EQUIPO en el frame táctico b + d.

  1. búsqueda gruesa: muchos frames del broadcast x miles de desfases -> desfases con más
     coincidencias (< 1.5 m); se espera uno por tiempo (el entretiempo está recortado)
  2. por ventanas de 10 s: el mejor desfase cercano (+-50 frames) a cada candidato; ventanas sin
     coincidencia (repeticiones, primeros planos, cortes) se descartan
  3. la pelota del broadcast en ventanas sincronizadas se pasa al reloj táctico

    .venv/bin/python solution/broadcast_sync.py
Salida: data/derived/solution/broadcast_ball.npz  (ball (N,2) normalizada en reloj táctico, NaN = sin dato)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, WORK, N_FRAMES, FPS, L_M, W_M, load_atd, team_of_each_track, halves  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

BRAW = os.path.join(ROOT, "data", "broadcast", "BAR-ATH_broadcast_atd.txt")
BMETA = os.path.join(ROOT, "data", "broadcast", "BAR-ATH_broadcast_atd.xml")
NB = 164488


def load_broadcast():
    cp = os.path.join(WORK, "broadcast_atd.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return z["players"], z["ball"], z["teams"]
    root = ET.parse(BMETA).getroot()
    home = [t.get("id") for t in root.iter("Team")][0]
    teams = np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)
    K = len(teams)
    pl = np.full((NB, K, 2), np.nan, np.float32)
    ball = np.full((NB, 2), np.nan, np.float32)
    with open(BRAW) as fh:
        for line in fh:
            fid, p_, b_ = line.rstrip("\n").split(":")
            i = int(fid) - 1
            for s, g in enumerate(p_.split(";")):
                if g[0] != "N":
                    x, y, _ = g.split(",")
                    pl[i, s] = (float(x), float(y))
            if b_[0] != "N":
                x, y, _ = b_.split(",")
                ball[i] = (float(x), float(y))
    np.savez_compressed(cp, players=pl, ball=ball, teams=teams)
    return pl, ball, teams


def frame_cost(bp, bt, tp_stack, tt):
    """bp (k,2) jugadores del broadcast con equipos bt; tp_stack (D,K,2) tácticos en D desfases."""
    cost = np.zeros(len(tp_stack))
    for team in (0, 1):
        b = bp[bt == team]
        if not len(b):
            continue
        t = tp_stack[:, tt == team]                                          # (D, Kt, 2)
        d = np.hypot(t[:, None, :, 0] - b[None, :, None, 0], t[:, None, :, 1] - b[None, :, None, 1])
        d = np.where(np.isnan(d), np.inf, d).min(2)                          # (D, kb)
        cost += np.minimum(d, 10).sum(1)
    return cost / len(bp)


def main():
    t0 = time.time()
    bpl, bball, bteams = load_broadcast()
    tpl, tball = load_atd()
    tteams = team_of_each_track()
    bm = bpl.astype(np.float64) * [L_M, W_M]
    tm = tpl.astype(np.float64) * [L_M, W_M]
    nvis = (~np.isnan(bm[..., 0])).sum(1)
    valid_b = np.flatnonzero(nvis >= 5)
    print(f"broadcast: {NB} frames, {len(valid_b)} con >= 5 jugadores ({time.time() - t0:.0f}s)", flush=True)

    # 1. búsqueda gruesa de desfases
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(valid_b, 400, replace=False))
    offs = np.arange(-NB, N_FRAMES, 25)
    votes = np.zeros(len(offs))
    for b in samp:
        tt_idx = b + offs
        ok = (tt_idx >= 0) & (tt_idx < N_FRAMES)
        bp = bm[b][~np.isnan(bm[b, :, 0])]
        bt = bteams[~np.isnan(bm[b, :, 0])]
        c = np.full(len(offs), np.inf)
        c[ok] = frame_cost(bp, bt, tm[tt_idx[ok]], tteams)
        votes += (c < 1.5)
    order = np.argsort(-votes)
    cands = []
    for i in order:
        if votes[i] < 5:
            break
        if all(abs(offs[i] - o) > 2000 for o in cands):
            cands.append(int(offs[i]))
        if len(cands) >= 4:
            break
    print("desfases candidatos (frame táctico = frame broadcast + d):",
          [(d, int(votes[offs == d][0])) for d in cands], f"({time.time() - t0:.0f}s)", flush=True)

    # 2. por ventanas de 10 s, el mejor desfase fino cerca de cada candidato
    W = 10 * FPS
    win_off = np.full(NB // W + 1, np.nan)
    win_cost = np.full(NB // W + 1, np.nan)
    fine = np.arange(-50, 51)
    for wi, w0 in enumerate(range(0, NB, W)):
        fr = [b for b in range(w0, min(NB, w0 + W), 5) if nvis[b] >= 4]
        if len(fr) < 10:
            continue
        best = (np.inf, None)
        for d0 in cands:
            tot = np.zeros(len(fine))
            cnt = 0
            for b in fr:
                ti = b + d0 + fine
                ok = (ti >= 0) & (ti < N_FRAMES)
                if not ok.all():
                    continue
                bp = bm[b][~np.isnan(bm[b, :, 0])]
                bt = bteams[~np.isnan(bm[b, :, 0])]
                tot += frame_cost(bp, bt, tm[ti], tteams)
                cnt += 1
            if cnt < 10:
                continue
            tot /= cnt
            k = int(np.argmin(tot))
            if tot[k] < best[0]:
                best = (tot[k], d0 + fine[k])
        if best[1] is not None:
            win_off[wi], win_cost[wi] = best[1], best[0]
    good = win_cost < 1.2
    print(f"ventanas de 10 s sincronizadas: {good.sum()} de {np.isfinite(win_cost).sum()} con jugadores "
          f"({time.time() - t0:.0f}s)", flush=True)
    vals, cnts = np.unique(win_off[good].astype(int), return_counts=True)
    top = sorted(zip(cnts, vals), reverse=True)[:8]
    print("desfases finos más frecuentes (ventanas, desfase):", top)

    # 3. pelota del broadcast al reloj táctico (solo ventanas sincronizadas)
    out = np.full((N_FRAMES, 2), np.nan, np.float32)
    for wi, w0 in enumerate(range(0, NB, W)):
        if not good[wi]:
            continue
        d = int(win_off[wi])
        b = np.arange(w0, min(NB, w0 + W))
        t = b + d
        ok = (t >= 0) & (t < N_FRAMES) & ~np.isnan(bball[b, 0])
        out[t[ok]] = bball[b[ok]]
    np.savez_compressed(os.path.join(WORK, "broadcast_ball.npz"), ball=out, win_off=win_off, win_cost=win_cost)
    g = np.zeros(N_FRAMES, bool)
    for _, a, b in halves():
        g[a:b + 1] = True
    has_t, has_b = ~np.isnan(tball[:, 0]) & g, ~np.isnan(out[:, 0]) & g
    both = has_t & has_b
    err = np.hypot(*((out[both] - tball[both]) * [L_M, W_M]).T)
    print(f"pelota del broadcast en tiempos calificados: {100 * has_b.sum() / g.sum():.1f}% | "
          f"rellena huecos del ATD táctico: {100 * (has_b & ~has_t).sum() / g.sum():.1f}% de los frames")
    print(f"donde ambos la ven: distancia broadcast-táctico mediana {np.median(err):.2f} m, p75 {np.percentile(err, 75):.2f}, "
          f"p90 {np.percentile(err, 90):.2f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
