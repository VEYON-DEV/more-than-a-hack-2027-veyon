"""Paso 2: trayectoria de la pelota con filtro de Kalman + suavizado RTS.

Estado [x, y, vx, vy] en metros del campo (modelo de velocidad constante con aceleración
aleatoria). En cada frame:
  1. predice;
  2. arma las mediciones posibles: candidatos del video (ball_candidates.npz) + la pelota del
     ATD si existe;
  3. se queda con las que caen dentro de la ventana de Mahalanobis y elige la de menor costo
     = distancia estadística + calidad del candidato (aislado, redondo, tamaño correcto);
  4. corrige con esa medición.
Sin mediciones durante LOST_FRAMES seguidos, la pelota se da por perdida (NaN) hasta que
aparece un ancla confiable (pelota del ATD o candidato de muy buena calidad).
Después, suavizado Rauch-Tung-Striebel hacia atrás en cada tramo seguido.

    .venv/bin/python solution/ball_track.py

Salida: data/derived/solution/ball_track.npz con, por frame de video:
  pos (N,2) m, vel (N,2) m/s, acc (N,2) m/s², std (N,) m, src (N,) 0 nada / 1 video / 2 ATD / 3 predicha
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, L_M, W_M, N_FRAMES, DT, halves, load_atd  # noqa: E402

SIGMA_A = 20.0        # m/s²: cuánto puede cambiar la velocidad (pases, rebotes)
SIGMA_VIDEO = 0.45    # m: error de un candidato del video proyectado al piso
SIGMA_ATD = 0.8       # m: error de la pelota del ATD
GATE_D2 = 16.0        # ventana de Mahalanobis (chi² 2 gl ~ 99.97%)
GATE_MAX_M = 6.0      # además, nunca más lejos que esto
LAMBDA_Q = 4.0        # peso de la calidad del candidato en el costo
Q_ATD = 0.6           # "calidad" asignada a la pelota del ATD
LOST_FRAMES = 12      # 0.5 s sin mediciones -> perdida (elegido con la prueba de esconder el ATD)
REINIT_Q = 1.0        # calidad mínima de un candidato del video para reanclar sin ATD
MAX_Q = 1.6           # candidatos del video peores que esto no se usan nunca. Medido sin
                      # etiquetas: el candidato que coincide con la pelota del ATD tiene calidad
                      # mediana 1.26; el conjunto de todos los candidatos, 3.36
SNAP_TO_VIDEO_M = 1.5 # con pelota del ATD, se usa un candidato bueno del video si está así de cerca

F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float64)
q1 = np.array([[DT ** 4 / 4, DT ** 3 / 2], [DT ** 3 / 2, DT ** 2]]) * SIGMA_A ** 2
Q = np.zeros((4, 4))
Q[np.ix_([0, 2], [0, 2])] = q1
Q[np.ix_([1, 3], [1, 3])] = q1


def candidate_quality(c):
    """Costo de 'no parecer pelota' (0 = pelota perfecta). Columnas de ball_candidates.npz."""
    ratio = np.clip(c[:, 6], 0.05, 20)
    fill, ring = c[:, 7], c[:, 9]
    return 3.0 * ring + 2.0 * (1.0 - np.clip(fill, 0, 1)) + 1.5 * np.abs(np.log(ratio))


def track_half(f0, f1, cand_by_frame, atd_ball_m, atd_sigma=None):
    n = f1 - f0 + 1
    xs_f = np.full((n, 4), np.nan)
    Ps_f = np.full((n, 4, 4), np.nan)
    xs_p = np.full((n, 4), np.nan)
    Ps_p = np.full((n, 4, 4), np.nan)
    src = np.zeros(n, np.int8)
    reinit = np.zeros(n, bool)
    x = None
    P = None
    since = 10 ** 9
    for i in range(n):
        f = f0 + i
        c = cand_by_frame.get(f)
        Zv = c[:, 3:5].astype(np.float64) if c is not None and len(c) else np.zeros((0, 2))
        qv = candidate_quality(c) if c is not None and len(c) else np.zeros(0)
        has_atd = not np.isnan(atd_ball_m[f, 0])
        za = atd_ball_m[f].astype(np.float64) if has_atd else None
        sig_a = SIGMA_ATD if atd_sigma is None else float(atd_sigma[f])

        if x is not None:
            x = F @ x
            P = F @ P @ F.T + Q
            xs_p[i], Ps_p[i] = x, P
        best = None
        good = qv < MAX_Q
        Zg, qg = Zv[good], qv[good]
        if has_atd:
            # manda el ATD; si hay un candidato bueno del video muy cerca, ese (es más preciso)
            if len(Zg):
                dist = np.hypot(Zg[:, 0] - za[0], Zg[:, 1] - za[1])
                k = int(np.argmin(dist + 0.3 * qg))
                if dist[k] < SNAP_TO_VIDEO_M:
                    best = (0.0, Zg[k], SIGMA_VIDEO ** 2, 1)
            if best is None:
                best = (0.0, za, sig_a ** 2, 2)
            if x is not None and since < LOST_FRAMES:
                r = best[1] - Hm @ x
                if np.hypot(*r) > GATE_MAX_M * 2:      # salto enorme: reanclar en el ATD
                    x = None
        elif x is not None and since < LOST_FRAMES and len(Zg):
            S_base = Hm @ P @ Hm.T
            Sinv = np.linalg.inv(S_base + np.eye(2) * SIGMA_VIDEO ** 2)
            r = Zg - Hm @ x
            d2 = np.einsum("ij,jk,ik->i", r, Sinv, r)
            ok = (d2 <= GATE_D2) & (np.hypot(r[:, 0], r[:, 1]) <= GATE_MAX_M)
            if ok.any():
                cost = np.where(ok, d2 + LAMBDA_Q * qg, np.inf)
                k = int(np.argmin(cost))
                best = (float(cost[k]), Zg[k], SIGMA_VIDEO ** 2, 1)
        if best is not None and x is None:
            z, R = best[1], best[2]
            x = np.array([z[0], z[1], 0.0, 0.0])
            P = np.diag([R, R, 15.0 ** 2, 15.0 ** 2])
            xs_p[i], Ps_p[i] = x.copy(), P.copy()
            reinit[i] = True
        if best is None and (x is None or since >= LOST_FRAMES):
            # (re)anclar: pelota del ATD, o el mejor candidato del video si es muy bueno
            anchor = None
            if has_atd:
                anchor = (za, sig_a ** 2, 2)
            elif len(qv) and qv.min() < REINIT_Q:
                k = int(np.argmin(qv))
                anchor = (Zv[k], SIGMA_VIDEO ** 2, 1)
            if anchor is not None:
                z, R, kind = anchor
                x = np.array([z[0], z[1], 0.0, 0.0])
                P = np.diag([R, R, 15.0 ** 2, 15.0 ** 2])
                xs_p[i], Ps_p[i] = x.copy(), P.copy()
                reinit[i] = True
                best = (0.0, z, R, kind)
        if best is not None:
            _, z, R, kind = best
            S = Hm @ P @ Hm.T + np.eye(2) * R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (z - Hm @ x)
            P = (np.eye(4) - K @ Hm) @ P
            since = 0
            src[i] = kind
        elif x is not None:
            since += 1
            if since < LOST_FRAMES:
                src[i] = 3
        if x is not None and since < LOST_FRAMES:
            xs_f[i], Ps_f[i] = x, P
    # suavizado RTS por tramos seguidos
    xs_s = xs_f.copy()
    Ps_s = Ps_f.copy()
    i = n - 1
    while i >= 0:
        if np.isnan(xs_f[i, 0]):
            i -= 1
            continue
        j = i
        while j - 1 >= 0 and not np.isnan(xs_f[j - 1, 0]) and not np.isnan(xs_p[j, 0]) and not reinit[j]:
            j -= 1
        for k in range(i - 1, j - 1, -1):          # tramo [j, i]
            Pp = Ps_p[k + 1]
            C = Ps_f[k] @ F.T @ np.linalg.inv(Pp)
            xs_s[k] = xs_f[k] + C @ (xs_s[k + 1] - xs_p[k + 1])
            Ps_s[k] = Ps_f[k] + C @ (Ps_s[k + 1] - Pp) @ C.T
        i = j - 1
    return xs_s, Ps_s, src


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(WORK, "ball_candidates.npz"))
    a = ap.parse_args()
    t = time.time()
    _, atd_ball = load_atd()
    atd_ball_m = (atd_ball * [L_M, W_M]).astype(np.float64)
    atd_sigma = np.full(len(atd_ball_m), SIGMA_ATD)
    bp = os.path.join(WORK, "broadcast_ball.npz")
    if os.path.exists(bp):
        # pelota del tracking del broadcast, sincronizada (broadcast_sync.py): solo donde el táctico no la ve
        bc = np.load(bp)["ball"].astype(np.float64) * [L_M, W_M]
        fill = np.isnan(atd_ball_m[:, 0]) & ~np.isnan(bc[:, 0])
        atd_ball_m[fill] = bc[fill]
        atd_sigma[fill] = 1.0
        print(f"pelota del broadcast usada en {fill.sum()} frames donde el táctico no la ve")
    d = np.load(a.cands)["data"]
    frames = d[:, 0].astype(np.int64)
    idx = np.flatnonzero(np.r_[True, np.diff(frames) != 0])
    cand_by_frame = {int(frames[a]): d[a:b] for a, b in zip(idx, np.r_[idx[1:], len(d)])}
    pos = np.full((N_FRAMES, 2), np.nan, np.float32)
    vel = np.full((N_FRAMES, 2), np.nan, np.float32)
    std = np.full(N_FRAMES, np.nan, np.float32)
    src = np.zeros(N_FRAMES, np.int8)
    for half, f0, f1 in halves():
        xs, Ps, s = track_half(f0, f1, cand_by_frame, atd_ball_m, atd_sigma)
        pos[f0:f1 + 1] = xs[:, :2]
        vel[f0:f1 + 1] = xs[:, 2:]
        std[f0:f1 + 1] = np.sqrt(np.maximum(Ps[:, 0, 0] + Ps[:, 1, 1], 0))
        src[f0:f1 + 1] = s
    acc = np.full_like(vel, np.nan)
    acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * DT)
    np.savez_compressed(os.path.join(WORK, "ball_track.npz"), pos=pos, vel=vel, acc=acc, std=std, src=src)
    graded = np.zeros(N_FRAMES, bool)
    for _, f0, f1 in halves():
        graded[f0:f1 + 1] = True
    g = graded
    known = ~np.isnan(pos[:, 0]) & g
    print(f"pelota conocida en {known.sum()}/{g.sum()} frames calificados ({100 * known.mean() / g.mean():.1f}%)")
    for k, name in ((1, "medición del video"), (2, "medición del ATD"), (3, "predicha (sin medición)")):
        print(f"   {name:24} {100 * ((src == k) & g).sum() / g.sum():5.1f}%")
    atd_has = ~np.isnan(atd_ball[:, 0]) & g
    print(f"   (el ATD solo trae pelota en {100 * atd_has.sum() / g.sum():.1f}%)")
    sp = np.hypot(*vel[known].T)
    print(f"velocidad: mediana {np.median(sp):.1f} m/s, p95 {np.percentile(sp, 95):.1f} m/s  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
