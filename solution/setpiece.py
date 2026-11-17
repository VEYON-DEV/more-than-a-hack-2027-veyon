"""Pistas de contexto de pelota parada / reanudación (se suman a features.py).

Una reanudación tiene una firma física: la pelota estuvo quieta un rato en un lugar típico de saque
(banderín de córner, línea de banda, área chica, punto central, punto penal) con los jugadores
acomodados (barrera a ~9 m, gente en el área), y de repente arranca.

  ball_still       frames seguidos con la pelota quieta (< 1 m/s) hasta ahora (tope 10 s)
  ball_still_fut   frames seguidos de pelota quieta desde ahora hacia adelante
  kick_from_rest   la pelota arranca (>= 5 m/s en 0.2 s) después de >= 1 s quieta, en +-0.5 s
  d_corner         distancia de la pelota al banderín más cercano (m)
  d_touch          distancia a la línea de banda más cercana (m)
  d_goalline       distancia a la línea de fondo más cercana (m)
  in_goal_area     pelota dentro de un área chica
  d_center         distancia al punto central (m)
  d_pen_spot       distancia al punto penal más cercano (m)
  n_9m_H / n_9m_A  jugadores de cada equipo a menos de 9.15 m de la pelota (barrera, distancia reglamentaria)
  n_box_H / _A     jugadores de cada equipo dentro de cada área grande
"""

import numpy as np

FPS = 25


def _run_length(flag):
    out = np.zeros(len(flag), np.float32)
    c = 0
    for i, f in enumerate(flag):
        c = c + 1 if f else 0
        out[i] = c
    return out


def setpiece_features(pm, teams, pos, vel, L=105.3, W=68.0):
    n = len(pos)
    known = ~np.isnan(pos[:, 0])
    sp = np.hypot(vel[:, 0], vel[:, 1])
    still = known & (np.nan_to_num(sp, nan=99) < 1.0)
    F = {}
    F["ball_still"] = np.minimum(_run_length(still), 250)
    F["ball_still_fut"] = np.minimum(_run_length(still[::-1])[::-1], 250)
    kick = np.zeros(n, bool)
    rs = _run_length(still)
    fast = np.nan_to_num(sp, nan=0) >= 5
    for i in np.flatnonzero(fast[5:]) + 5:
        if rs[i - 5] >= FPS:
            kick[i] = True
    F["kick_from_rest"] = np.convolve(kick.astype(float), np.ones(25), "same")
    x, y = pos[:, 0], pos[:, 1]
    cx = np.minimum(np.abs(x), np.abs(L - x))
    cy = np.minimum(np.abs(y), np.abs(W - y))
    F["d_corner"] = np.hypot(cx, cy)
    F["d_touch"] = cy
    F["d_goalline"] = cx
    F["in_goal_area"] = ((cx < 5.5) & (np.abs(y - W / 2) < 9.16)).astype(np.float32)
    F["d_center"] = np.hypot(x - L / 2, y - W / 2)
    F["d_pen_spot"] = np.minimum(np.hypot(x - 11, y - W / 2), np.hypot(x - (L - 11), y - W / 2))
    d = np.hypot(pm[..., 0] - x[:, None], pm[..., 1] - y[:, None])
    inbox = ((pm[..., 0] < 16.5) | (pm[..., 0] > L - 16.5)) & (np.abs(pm[..., 1] - W / 2) < 20.16)
    for t, nm in ((0, "H"), (1, "A")):
        sel = teams[None] == t
        F[f"n_9m_{nm}"] = np.where(known, np.nansum(np.where(sel, (d < 9.15).astype(float), np.nan), axis=1), np.nan)
        F[f"n_box_{nm}"] = np.nansum(np.where(sel, inbox.astype(float), np.nan), axis=1)
    for k in ("d_corner", "d_touch", "d_goalline", "d_center", "d_pen_spot", "in_goal_area"):
        F[k] = np.where(known, F[k], np.nan)
    return F
