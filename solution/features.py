"""Pistas por frame para decidir la posesión. El MISMO código para IDSSE y para Barça-Athletic.

Entrada (un tiempo de partido):
  players (n, K, 2) metros, NaN = ausente      teams (K,) 0 local / 1 visitante
  gk      (K,) bool                             pos, vel (n, 2) pelota en m y m/s (NaN = desconocida)
Salida: matriz (n, F) con NaN donde una pista no existe (el gradient boosting los acepta).
"""

import os

import numpy as np

FPS = 25


def movavg(x, w):
    v = np.where(np.isnan(x), 0, x)
    m = (~np.isnan(x)).astype(float)
    k = np.ones(w)
    num = np.convolve(v, k, "same")
    den = np.convolve(m, k, "same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def window(x, a, b):
    """Media de x en [t+a, t+b] (a, b en frames, pueden ser negativos), ignorando NaN."""
    n = len(x)
    v = np.where(np.isnan(x), 0, x)
    m = (~np.isnan(x)).astype(float)
    cv = np.r_[0, np.cumsum(v)]
    cm = np.r_[0, np.cumsum(m)]
    i0 = np.clip(np.arange(n) + a, 0, n)
    i1 = np.clip(np.arange(n) + b + 1, 0, n)
    s, c = cv[i1] - cv[i0], cm[i1] - cm[i0]
    return np.where(c > 0, s / np.maximum(c, 1e-9), np.nan)


def frozen_mask(p):
    still = np.zeros(p.shape[:2], bool)
    still[1:] = np.abs(np.diff(p, axis=0)).sum(-1) < 1e-4
    out = np.zeros_like(still)
    for s in range(p.shape[1]):
        e = np.flatnonzero(np.diff(np.r_[0, still[:, s].astype(int), 0]))
        for a, b in zip(e[::2], e[1::2]):
            if b - a >= FPS:
                out[a:b, s] = True
    return out


def noisy_players(pm, sigma, seed):
    """Ruido de prueba: el tracking de Barça-Athletic es más suave que el de IDSSE ensuciado con el que se
    entrena (18 % de jugador-frames casi quietos contra 8 %). Se le suma el mismo ruido suave en el tiempo
    que bench.degrade (sigma m, media móvil de 1 s), con semilla fija, para que se parezca al de
    entrenamiento. Los jugadores congelados se marcan como ausentes antes, igual que en features()."""
    pm = pm.astype(np.float64).copy()
    pm[frozen_mask(pm)] = np.nan
    if sigma <= 0:
        return pm
    wn = np.random.default_rng(seed).normal(0, 1, pm.shape)
    sm = np.apply_along_axis(lambda x: np.convolve(x, np.ones(FPS) / FPS, "same"), 0, wn)
    return pm + sm * (sigma / max(sm.std(), 1e-9))


def time_since(flag):
    """Frames desde el último True (inf si nunca)."""
    n = len(flag)
    last = np.where(flag, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(last)
    return np.arange(n) - last


NAMES = []


def features(players, teams, gk, pos, vel, L=105.3, W=68.0):
    n, K, _ = players.shape
    pm = players.astype(np.float64).copy()
    pm[frozen_mask(pm)] = np.nan
    pos = pos.astype(np.float64)
    vel = vel.astype(np.float64)
    known = ~np.isnan(pos[:, 0])
    F = {}

    # --- distancias pelota-jugadores por equipo
    d = np.hypot(pm[..., 0] - pos[:, None, 0], pm[..., 1] - pos[:, None, 1])
    for t, nm in ((0, "H"), (1, "A")):
        dt = np.where(teams[None] == t, d, np.nan)
        srt = np.sort(np.where(np.isnan(dt), np.inf, dt), axis=1)
        F[f"d1{nm}"] = np.where(np.isfinite(srt[:, 0]), srt[:, 0], np.nan)
        F[f"d2{nm}"] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)
        F[f"n2m{nm}"] = np.where(known, (dt < 2).sum(1), np.nan)
        F[f"n5m{nm}"] = np.where(known, (dt < 5).sum(1), np.nan)
    F["ddiff"] = F["d1A"] - F["d1H"]                       # >0: el local está más cerca
    for a, b, tag in ((-25, 0, "p1"), (0, 25, "f1"), (-50, -26, "p2"), (26, 50, "f2"), (-12, 12, "c")):
        F[f"ddiff_{tag}"] = window(F["ddiff"], a, b)

    # --- control: a <1 m del más cercano y moviéndose con él
    dd = np.where(np.isnan(d), np.inf, d)
    near = np.argmin(dd, axis=1)
    dnear = dd[np.arange(n), near]
    pv = np.vstack([np.zeros((1, K, 2)), np.diff(pm, axis=0) * FPS])
    pvn = pv[np.arange(n), near]
    pvn = np.stack([movavg(pvn[:, 0], 5), movavg(pvn[:, 1], 5)], 1)
    rel = np.hypot(*(vel - pvn).T)
    F["rel_near"] = np.where(np.isfinite(dnear), rel, np.nan)
    ctrl = known & (dnear < 1.0) & (np.nan_to_num(rel, nan=99) < 3.0)
    cH = (ctrl & (teams[near] == 0)).astype(float)
    cA = (ctrl & (teams[near] == 1)).astype(float)
    for a, b, tag in ((-3, 3, "c"), (-25, 0, "p1"), (0, 25, "f1"), (-75, -26, "p3"), (26, 75, "f3")):
        F[f"ctrlH_{tag}"] = window(cH, a, b)
        F[f"ctrlA_{tag}"] = window(cA, a, b)
    # equipo del último / próximo control y cuánto falta
    lastH, lastA = time_since(cH > 0), time_since(cA > 0)
    nextH, nextA = time_since(cH[::-1] > 0)[::-1], time_since(cA[::-1] > 0)[::-1]
    F["since_ctrl_diff"] = np.clip(lastA, 0, 500) - np.clip(lastH, 0, 500)     # >0: el local controló más reciente
    F["until_ctrl_diff"] = np.clip(nextA, 0, 500) - np.clip(nextH, 0, 500)
    F["since_ctrl_any"] = np.clip(np.minimum(lastH, lastA), 0, 500)
    F["until_ctrl_any"] = np.clip(np.minimum(nextH, nextA), 0, 500)

    # --- pelota: velocidad, toques, afuera, posición
    sp = np.hypot(vel[:, 0], vel[:, 1])
    F["bspeed"] = movavg(sp, 5)
    F["bspeed_p1"] = window(sp, -25, 0)
    F["bspeed_f1"] = window(sp, 0, 25)
    vb = np.stack([window(vel[:, k], -6, -2) for k in range(2)], 1)
    va = np.stack([window(vel[:, k], 2, 6) for k in range(2)], 1)
    F["dv"] = np.hypot(*(va - vb).T)
    out_d = np.maximum.reduce([-pos[:, 0], pos[:, 0] - L, -pos[:, 1], pos[:, 1] - W])   # >0 fuera
    F["out_dist"] = out_d
    F["out_p1"] = window((out_d > 0.3).astype(float) * np.where(known, 1, np.nan), -25, 0)
    F["out_since"] = np.clip(time_since(np.nan_to_num(out_d, nan=-1) > 0.3), 0, 1500)
    # dirección: dónde defiende el local (por sus arqueros, o por su centroide)
    gx = np.nanmean(np.where((teams == 0) & gk, pm[..., 0], np.nan)) if (gk & (teams == 0)).any() else np.nan
    if np.isnan(gx):
        gx = np.nanmean(np.where(teams[None] == 0, pm[..., 0], np.nan))
    sgn = 1.0 if gx < L / 2 else -1.0                      # +1: el local defiende el lado x=0
    F["bx_home"] = sgn * (pos[:, 0] - L / 2) / (L / 2)       # +1 cerca del arco rival del local
    F["by_center"] = np.abs(pos[:, 1] - W / 2) / (W / 2)
    for t, nm in ((0, "H"), (1, "A")):
        cx = np.nanmean(np.where(teams[None] == t, pm[..., 0], np.nan), axis=1)
        F[f"cx{nm}"] = sgn * (cx - L / 2) / (L / 2)
        F[f"spread{nm}"] = np.nanstd(np.where(teams[None] == t, pm[..., 0], np.nan), axis=1)
    F["cx_diff"] = F["cxH"] - F["cxA"]

    # --- dinámica de equipo: quién avanza, quién se repliega, quién presiona
    xrel = sgn * (pm[..., 0] - L / 2) / (L / 2)            # orientado: el local ataca hacia +1
    for t, nm in ((0, "H"), (1, "A")):
        c = movavg(np.nanmean(np.where(teams[None] == t, xrel, np.nan), axis=1), FPS)
        v = np.gradient(c) * FPS * (L / 2)                  # m/s hacia el arco rival del local
        F[f"vcx{nm}"] = v if t == 0 else -v                 # cada equipo, hacia su arco rival
        F[f"spready{nm}"] = np.nanstd(np.where(teams[None] == t, pm[..., 1], np.nan), axis=1)
        F[f"area{nm}"] = F[f"spread{nm}"] * F[f"spready{nm}"]
    F["vcx_diff"] = F["vcxH"] - F["vcxA"]
    F["area_diff"] = F["areaH"] - F["areaA"]
    bxh = F["bx_home"][:, None]
    behH = np.where(known, np.nansum(np.where(teams[None] == 0, (xrel < bxh).astype(float), np.nan), axis=1), np.nan)
    behA = np.where(known, np.nansum(np.where(teams[None] == 1, (xrel > bxh).astype(float), np.nan), axis=1), np.nan)
    F["behindH"], F["behindA"] = behH, behA
    F["behind_diff"] = behA - behH                          # >0: más visitantes detrás -> ataca el local
    # presión: rivales a <3 m del jugador de cada equipo más cercano a la pelota
    for t, nm in ((0, "H"), (1, "A")):
        dt = np.where(teams[None] == t, dd, np.inf)
        j = np.argmin(dt, axis=1)
        pj = pm[np.arange(n), j]                            # (n, 2)
        dop = np.hypot(pm[..., 0] - pj[:, None, 0], pm[..., 1] - pj[:, None, 1])
        F[f"press{nm}"] = np.where(np.isfinite(dt[np.arange(n), j]),
                                   np.nansum(np.where(teams[None] != t, (dop < 3).astype(float), np.nan), axis=1),
                                   np.nan)
    F["press_diff"] = F["pressH"] - F["pressA"]

    # --- vectores jugador -> pelota: quién corre hacia la pelota, quién llega primero
    pvs = np.stack([np.apply_along_axis(lambda x: movavg(x, 5), 0, pv[..., k]) for k in range(2)], -1)  # (n,K,2)
    to_b = np.stack([pos[:, None, 0] - pm[..., 0], pos[:, None, 1] - pm[..., 1]], -1)
    dist = np.hypot(to_b[..., 0], to_b[..., 1])
    unit = to_b / np.maximum(dist[..., None], 1e-6)
    close = (pvs * unit).sum(-1)                              # m/s hacia la pelota (+) o alejándose (-)
    near15 = dist < 15
    t_arr = dist / np.maximum(close + 3.0, 1.0)               # tiempo aproximado para llegar a la pelota
    for t, nm in ((0, "H"), (1, "A")):
        sel = (teams[None] == t) & near15
        F[f"close{nm}"] = np.where(known, np.nanmean(np.where(sel, close, np.nan), axis=1), np.nan)
        F[f"chasers{nm}"] = np.where(known, np.nansum(np.where(sel, (close > 2).astype(float), np.nan), axis=1), np.nan)
        tt = np.where(teams[None] == t, t_arr, np.nan)
        F[f"tarrive{nm}"] = np.where(known, np.nanmin(np.where(np.isnan(tt), np.inf, tt), axis=1), np.nan)
    F["close_diff"] = F["closeH"] - F["closeA"]
    F["chasers_diff"] = F["chasersH"] - F["chasersA"]
    F["tarrive_diff"] = F["tarriveA"] - F["tarriveH"]          # >0: el local llega antes
    # dirección de la pelota respecto del ataque del local (pases hacia adelante = posesión)
    F["bvx_home"] = sgn * vel[:, 0]
    F["bvx_home_p1"] = window(sgn * vel[:, 0], -25, 0)
    F["bvx_home_f1"] = window(sgn * vel[:, 0], 0, 25)

    # --- ritmo del juego: en una parada caminan
    ps = np.hypot(*np.diff(pm, axis=0).transpose(2, 0, 1)) * FPS
    ps = np.vstack([np.full((1, K), np.nan), ps])
    ps[ps > 12] = np.nan
    med = np.nanmedian(ps, axis=1)
    F["pspeed"] = movavg(med, FPS)
    F["pspeed_p2"] = window(med, -50, 0)
    F["pspeed_f2"] = window(med, 0, 50)
    F["n_players"] = (~np.isnan(pm[..., 0])).sum(1).astype(float)

    # --- pelota conocida / huecos
    F["known"] = known.astype(float)
    F["since_known"] = np.clip(time_since(known), 0, 500)
    F["until_known"] = np.clip(time_since(known[::-1])[::-1], 0, 500)
    F["known_p2"] = window(known.astype(float), -50, 50)

    # --- contexto de pelota parada / reanudación (setpiece.py): EXPERIMENTAL, apagado por defecto
    #     hasta validarlo en el banco (SETPIECE=1 para encenderlo)
    # --- toques rápidos (TOUCH=1): cambio brusco de la velocidad de la pelota junto a un jugador.
    #     Captura pases de primera, intercepciones y desvíos, donde la pelota no "se queda" con nadie.
    if os.environ.get("TOUCH") == "1":
        vf = np.stack([np.nan_to_num(np.r_[vel[3:, k], np.full(3, np.nan)]) for k in range(2)], 1)
        vp = np.stack([np.nan_to_num(np.r_[np.full(3, np.nan), vel[:-3, k]]) for k in range(2)], 1)
        jump = np.hypot(*(vf - vp).T)
        jump[~known] = 0
        peak = (jump > 3.0) & (jump >= np.r_[0, jump[:-1]]) & (jump >= np.r_[jump[1:], 0])
        tch = peak & (dnear < 2.0)
        tH = (tch & (teams[near] == 0)).astype(float)
        tA = (tch & (teams[near] == 1)).astype(float)
        lH, lA = time_since(tH > 0), time_since(tA > 0)
        nH, nA = time_since(tH[::-1] > 0)[::-1], time_since(tA[::-1] > 0)[::-1]
        F["since_touch_diff"] = np.clip(lA, 0, 500) - np.clip(lH, 0, 500)      # >0: el local tocó más reciente
        F["until_touch_diff"] = np.clip(nA, 0, 500) - np.clip(nH, 0, 500)
        F["since_touch_any"] = np.clip(np.minimum(lH, lA), 0, 500)
        F["touchH_p1"], F["touchA_p1"] = window(tH, -25, 0) * 25, window(tA, -25, 0) * 25
        F["touchH_f1"], F["touchA_f1"] = window(tH, 0, 25) * 25, window(tA, 0, 25) * 25
        F["ball_jump"] = jump
    if os.environ.get("SETPIECE") == "1":
        from setpiece import setpiece_features
        F.update(setpiece_features(pm, teams, pos, vel, L, W))

    global NAMES
    NAMES = list(F.keys())
    with np.errstate(all="ignore"):
        X = np.stack([np.asarray(F[k], np.float64) for k in NAMES], axis=1).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X
