"""Banco de prueba en IDSSE (7 partidos de la Bundesliga, otros partidos -> permitido).

Por partido: ensucia el tracking como el nuestro, pasa la pelota por el mismo Kalman, calcula las
mismas pistas y compara métodos con el evaluador OFICIAL (evaluation/possession_metrics.py),
dejando un partido afuera cada vez (se entrena con 6 y se mide en el 7.º).

Etiquetas: BallStatus/BallPossession de la DFL convertidas a "posesión controlada" (definición de
la competencia: un toque sin control no cambia la posesión): un tramo de un equipo más corto que
TAU_CTRL no cuenta, la posesión sigue con el equipo anterior.

    .venv/bin/python solution/bench.py
"""

import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as pm  # noqa: E402
from common import WORK, BENCH_SUB, FPS, DEGRADE  # noqa: E402
from features import features, window  # noqa: E402
import features as featmod  # noqa: E402
import ball_track  # noqa: E402

TAU_CTRL = 3.0           # s: mínimo para que un cambio de posesión cuente como control
CACHE = os.path.join(WORK, BENCH_SUB)
os.makedirs(CACHE, exist_ok=True)


def controlled(lab, tau_f):
    lab = lab.copy()
    for _ in range(3):
        cuts = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        st, en = np.r_[0, cuts], np.r_[cuts, len(lab)]
        changed = False
        for i, (s, e) in enumerate(zip(st, en)):
            if lab[s] == 2 or e - s >= tau_f:
                continue
            prev = lab[st[i - 1]] if i > 0 else 2
            nxt = lab[st[i + 1]] if i + 1 < len(st) else 2
            if prev != 2:
                lab[s:e] = prev
                changed = True
            elif nxt != 2 and nxt != lab[s]:
                lab[s:e] = nxt
                changed = True
        if not changed:
            break
    return lab


def renewal_mask(n, on_lengths, off_lengths, rng, start_on=True):
    """True/False alternando con largos muestreados de distribuciones empíricas (en frames)."""
    m = np.zeros(n, bool)
    i, on = 0, start_on
    while i < n:
        L = int(rng.choice(on_lengths if on else off_lengths))
        m[i:i + max(L, 1)] = on
        i += max(L, 1)
        on = not on
    return m


def smooth_players(p, w):
    """Media móvil de w frames por jugador y coordenada, sobre los tramos presentes (bordes repetidos)."""
    out = p.copy()
    k = np.ones(w) / w
    for j in range(p.shape[1]):
        for c in range(2):
            x = p[:, j, c]
            ok = ~np.isnan(x)
            if ok.sum() > w:
                out[ok, j, c] = np.convolve(np.pad(x[ok], (w // 2, w - 1 - w // 2), mode="edge"), k, "valid")
    return out


def degrade(players, ball, rng, live=None, pitch=None, teams=None, gk=None):
    """Ensucia el tracking de IDSSE con las estadísticas medidas en nuestro tracking.
    live (n,) bool: juego vivo por frame; con DEGRADE=v2 los huecos de la pelota dependen de él."""
    n, K, _ = players.shape
    if DEGRADE != "v1" and live is not None:
        gs = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ball_gaps_by_state.npz"))
        # v4: dentro del campo la pelota muerta se ve más (55 %) y fuera casi nunca (abajo), para que en total
        # se vea el 43 % del tiempo muerto con solo ~17 % de eso fuera de las líneas, como en nuestro tracking
        gap_dead = (gs["gap_dead"] * 0.2).astype(int) + 1 if DEGRADE.startswith("v4") else gs["gap_dead"]
        vis = np.where(live, renewal_mask(n, gs["vis_live"], gs["gap_live"], rng),
                       renewal_mask(n, gs["vis_dead"], gap_dead, rng, start_on=False))
    else:
        st = np.load(os.path.join(WORK, "final_ball_gap_stats.npz"))
        vis = renewal_mask(n, st["vis"], st["gaps"], rng)
    if DEGRADE.startswith("v3r"):
        # v3r: error real de nuestro detector de pelota en el video. En Barça-Athletic el 45 % de la pelota
        # conocida viene del video, en rachas cortas entre mediciones del proveedor (ball_src_runs.npz), y ahí
        # el error es mediana 0.68 m, 82 % < 2 m, p90 3.4 m (prueba de esconder el tracking del proveedor).
        # Se reproduce con temblor de 0.7 m y, en el 30 % de las rachas, un salto de ~3 m (otro objeto blanco).
        rs = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ball_src_runs.npz"))
        vm = renewal_mask(n, rs["vid"], rs["oth"], rng, start_on=False)
        nz = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, "same"), 0, rng.normal(0, 1, (n, 2)))
        nz *= 0.7 / max(nz.std(), 1e-9)
        off = np.zeros((n, 2))
        ev = np.flatnonzero(np.diff(np.r_[0, vm.astype(int), 0]))
        for a0, a1 in zip(ev[::2], ev[1::2]):
            if rng.random() < 0.3:
                ang, r = rng.uniform(0, 2 * np.pi), rng.lognormal(np.log(3.0), 0.7)
                off[a0:a1] = [r * np.cos(ang), r * np.sin(ang)]
        vid_err = (vm, nz + off)
    else:
        vid_err = None
    if DEGRADE.startswith("v4") and pitch is not None:
        # v4: con la pelota fuera de las líneas nuestro tracking casi nunca la tiene (el proveedor deja de
        # seguirla): fuera de las líneas 4.9 % de la pelota conocida en Barça-Athletic contra 14.6 % en v3.
        # Se borra la pelota en el 85 % de los tramos en que está afuera (+-0.5 s).
        Lp, Wp = pitch
        out = ~np.isnan(ball[:, 0]) & ((ball[:, 0] < 0) | (ball[:, 0] > Lp) | (ball[:, 1] < 0) | (ball[:, 1] > Wp))
        out = np.convolve(out.astype(float), np.ones(25), "same") > 0      # +-0.5 s: también la salida y la vuelta
        e = np.flatnonzero(np.diff(np.r_[0, out.astype(int), 0]))
        for a0, a1 in zip(e[::2], e[1::2]):
            if rng.random() < 0.85:
                vis[a0:a1] &= live[a0:a1] if live is not None else False     # solo la parte con pelota muerta
    bn = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5) / 5, "same"), 0, rng.normal(0, 1, ball.shape))
    # v3p (experimento "pelota perfecta"): sin ruido ni pelota pegada a los pies, mismos huecos
    perfect = DEGRADE.startswith("v3p")
    b = ball.astype(np.float64) + (0.0 if perfect else 1.0) * bn * (0.3 / max(bn.std(), 1e-9))
    b[~vis] = np.nan
    # errores de nuestro detector en el video: a veces la pelota "se pega" unos frames a los pies
    # del jugador más cercano (medias, botines). Se simula en ~30 % de los tramos visibles.
    e = np.flatnonzero(np.diff(np.r_[0, vis.astype(int), 0]))
    for a0, a1 in zip(e[::2], e[1::2]):
        if perfect or a1 - a0 < 10 or rng.random() > 0.3:
            continue
        L_ = int(rng.integers(5, 26))
        s0 = int(rng.integers(a0, max(a0 + 1, a1 - L_)))
        bb = ball[s0]
        if np.isnan(bb[0]):
            continue
        dpl = np.hypot(players[s0, :, 0] - bb[0], players[s0, :, 1] - bb[1])
        if np.all(np.isnan(dpl)) or np.nanmin(dpl) > 4:
            continue
        j = int(np.nanargmin(dpl))
        seg = players[s0:s0 + L_, j].astype(np.float64)
        ok = ~np.isnan(seg[:, 0])
        b[s0:s0 + L_][ok] = seg[ok] + rng.normal(0, 0.3, 2)
    if vid_err is not None:
        vm, err = vid_err
        b[vm] += err[vm]
    # ruido de posición SUAVE en el tiempo (como un tracker real): desplaza ~0.15 m pero no inventa
    # velocidad. (Un ruido independiente por frame hacía 'correr' a todos a ~6 m/s.)
    if DEGRADE.startswith("v3"):
        # v3: nuestro tracking es MÁS suave que el de IDSSE (18 % de jugador-frames casi quietos contra
        # 11-13 % en IDSSE limpio y 8 % con el ruido de v1/v2). En vez de sumar ruido, se suaviza 1 s.
        p = smooth_players(players.astype(np.float64), FPS)
        if "j" in DEGRADE[2:]:
            # v3rj: saltos de identidad como en nuestro tracking (el punto de un jugador pasa a seguir a otro
            # compañero y se queda ahí). Barça-Athletic: 0.35-1.0 saltos > 0.5 m por jugador-minuto.
            # grandes (cambio de identidad, 1-10 m): ~0.13 por jugador-minuto
            n_ev = rng.poisson(0.13 * K / 2 * n / FPS / 60)
            # chicos (0.5-1 m, el punto se corre y vuelve a los pocos segundos): ~0.5 por jugador-minuto
            n_sm = rng.poisson(0.5 * K / 2 * n / FPS / 60)
            for t in rng.integers(1, n - 1, n_sm):
                k = int(rng.integers(0, K))
                dur = int(rng.exponential(3 * FPS)) + 1
                ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.0)
                p[t:t + dur, k] += [r * np.cos(ang), r * np.sin(ang)]
            for t in np.sort(rng.integers(1, n - 1, n_ev)):
                pres = np.flatnonzero(~np.isnan(p[t, :, 0]))
                if len(pres) < 2:
                    continue
                k = int(rng.choice(pres))
                dd = np.hypot(*(p[t, pres] - p[t, k]).T)
                same = (teams[pres] == teams[k]) if teams is not None else np.ones(len(pres), bool)
                if gk is not None:            # nunca el arquero: features() orienta la cancha con su posición
                    if gk[k]:
                        continue
                    same &= ~gk[pres]
                cand = pres[(dd > 1.0) & (dd < 10) & same]       # entre compañeros: el color corrige el equipo
                if len(cand) == 0:
                    continue
                j = int(rng.choice(cand))
                p[t:, [k, j]] = p[t:, [j, k]]
    else:
        wn = rng.normal(0, 1, players.shape)
        k = np.ones(FPS) / FPS
        sm = np.apply_along_axis(lambda x: np.convolve(x, k, "same"), 0, wn)
        sm *= 0.15 / max(sm.std(), 1e-9)
        p = players.astype(np.float64) + sm
    teams_flip = np.zeros((n, K), bool)
    # v4: nuestro tracking pierde menos jugadores sueltos (22 presentes el 47 % del tiempo contra 24 % en v3)
    # pero a veces pierde varios a la vez (14-18 presentes el 5 % del tiempo)
    on_mean = 266 * (2.2 if DEGRADE.startswith("v4") else 1.0)
    for k in range(K):
        on = rng.exponential(on_mean * FPS, 200).astype(int) + 1
        off = (rng.lognormal(np.log(7.5 * FPS), 1.2, 200)).astype(int) + 1
        miss = ~renewal_mask(n, on, off, rng, start_on=bool(rng.random() > 0.05))
        p[miss, k] = np.nan
        on = rng.exponential(500 * FPS, 200).astype(int) + 1
        off = (rng.lognormal(np.log(8 * FPS), 0.8, 200)).astype(int) + 1
        teams_flip[:, k] = ~renewal_mask(n, on, off, rng, start_on=True)
    if DEGRADE.startswith("v4"):
        t = int(rng.exponential(40 * FPS))
        while t < n:
            d = int(rng.lognormal(np.log(1.5 * FPS), 0.6)) + 1
            pres = np.flatnonzero(~np.isnan(p[t, :, 0]))
            if len(pres) > 8:
                p[t:t + d, rng.choice(pres, int(rng.integers(3, 9)), replace=False)] = np.nan
            t += d + int(rng.exponential(40 * FPS))
    return p, b, teams_flip


def swap_labels(y):
    y = y.copy()
    h, a = y == 0, y == 1
    y[h], y[a] = 1, 0
    return y


def match_features(path, seed, swap=False):
    suffix = f"_s{seed}w.npz" if swap else f"_s{seed}.npz"
    cp = os.path.join(CACHE, os.path.basename(path).replace(".npz", suffix))
    if os.path.exists(cp):
        z = np.load(cp)
        featmod.NAMES = list(z["names"])
        return [z["X0"], z["X1"]], [z["y0"], z["y1"]], [z["raw0"], z["raw1"]]
    z = np.load(path)
    rng = np.random.default_rng(seed)
    L, W = z["pitch"]
    Xs, ys, raws = [], [], []
    for h in (0, 1):
        if seed == 0:      # techo: tracking perfecto (sin ensuciar)
            p, b = z[f"players_{h}"].astype(np.float64), z[f"ball_{h}"].astype(np.float64)
            flip = np.zeros(p.shape[:2], bool)
        else:
            live = controlled(z[f"label_{h}"], int(1.5 * FPS)) != 2      # solo lo usa DEGRADE=v2
            p, b, flip = degrade(z[f"players_{h}"], z[f"ball_{h}"], rng, live=live, pitch=(L, W), teams=z["teams"],
                                 gk=z["gk"])
        n = len(b)
        xs, _, _ = ball_track.track_half(0, n - 1, {}, b)
        pos, vel = xs[:, :2], xs[:, 2:]
        teams = z["teams"] if not swap else (1 - z["teams"]).astype(np.int8)
        # equipos cambiados por tramos: se simula intercambiando las posiciones entre 'equipos'
        # (sin identidades, equivale a que el tracker diga el equipo equivocado)
        Kp = p.shape[1]
        pf = np.concatenate([np.where(flip[..., None], np.nan, p), np.where(flip[..., None], p, np.nan)], axis=1)
        tf = np.r_[teams, 1 - teams]
        gkf = np.r_[z["gk"], z["gk"]]
        X = features(pf, tf, gkf, pos, vel, L, W)
        Xs.append(X)
        lab_h = swap_labels(z[f"label_{h}"]) if swap else z[f"label_{h}"]
        ys.append(controlled(lab_h, int(TAU_CTRL * FPS)))
        raws.append(lab_h)
    np.savez_compressed(cp, X0=Xs[0], X1=Xs[1], y0=ys[0], y1=ys[1], raw0=raws[0], raw1=raws[1],
                        names=np.array(featmod.NAMES))
    return Xs, ys, raws


def to_codes(halves_labels):
    """Concatena dos tiempos con un frame X en medio (así el evaluador no los mezcla)."""
    return np.r_[halves_labels[0], 3, halves_labels[1]].astype(np.int8)


def score(gt_halves, pred_halves):
    r = pm.evaluate(to_codes(gt_halves), to_codes(pred_halves), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def majority(lab, w):
    out = np.empty_like(lab)
    oh = np.stack([(lab == k).astype(float) for k in range(3)], 1)
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(oh, 0)])
    n = len(lab)
    i0 = np.clip(np.arange(n) - w, 0, n)
    i1 = np.clip(np.arange(n) + w + 1, 0, n)
    return np.argmax(cs[i1] - cs[i0], axis=1).astype(np.int8)


def baseline(X):
    i = featmod.NAMES.index
    known = X[:, i("known")] > 0
    lab = np.where(known, np.where(np.nan_to_num(X[:, i("ddiff")], nan=0) > 0, 0, 1), 2).astype(np.int8)
    return majority(lab, 50)


def transitions_from(ys):
    C = np.ones((3, 3))
    for y in ys:
        np.add.at(C, (y[:-1], y[1:]), 1)
    return C / C.sum(1, keepdims=True)


try:
    from numba import njit
except ImportError:                      # sin numba funciona igual, solo más lento
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


@njit(cache=True)
def _viterbi(logp, logT):
    n = logp.shape[0]
    score = logp[0].copy()
    back = np.zeros((n, 3), np.int8)
    new = np.empty(3)
    for t in range(1, n):
        for j in range(3):
            best, arg = -1e300, 0
            for i in range(3):
                v = score[i] + logT[i, j]
                if v > best:
                    best, arg = v, i
            back[t, j] = arg
            new[j] = best + logp[t, j]
        score[:] = new
    path = np.zeros(n, np.int8)
    path[n - 1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi(logp, logT):
    return _viterbi(np.ascontiguousarray(logp, np.float64), np.ascontiguousarray(logT, np.float64))


def adjust_T(T, alpha=1.0, beta=1.0):
    """alpha escala los cambios H<->A; beta los cambios vivo<->muerto."""
    T2 = T.copy()
    T2[0, 1] *= alpha
    T2[1, 0] *= alpha
    T2[0, 2] *= beta
    T2[1, 2] *= beta
    T2[2, 0] *= beta
    T2[2, 1] *= beta
    for i in range(3):
        off = T2[i].sum() - T2[i, i]
        T2[i, i] = max(1e-6, 1 - off)
    return T2 / T2.sum(1, keepdims=True)


def hmm_decode(proba, prior, T, tau, alpha=1.0, beta=1.0):
    loglik = np.log(np.clip(proba, 1e-6, 1)) - np.log(prior)
    return viterbi(tau * loglik, np.log(adjust_T(T, alpha, beta)))


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=48,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def main():
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz")))
    data = {}
    for p in paths:
        mid = os.path.basename(p)[:-4]
        data[mid] = match_features(p, seed=1)
        print(f"  pistas {mid}: {sum(len(x) for x in data[mid][0])} frames ({time.time() - t0:.0f}s)", flush=True)
    names = featmod.NAMES
    print(f"{len(names)} pistas por frame")
    taus = (0.05, 0.1, 0.2, 0.35, 0.6, 1.0)
    res = {"baseline": [], "clasificador": [], **{f"clf+markov tau={t}": [] for t in taus}}
    probas = {}
    for mid in data:
        train = [m for m in data if m != mid]
        Xtr = np.vstack([x[::3] for m in train for x in data[m][0]])
        ytr = np.concatenate([y[::3] for m in train for y in data[m][1]])
        model = make_model().fit(Xtr, ytr)
        prior = np.bincount(ytr, minlength=3) / len(ytr)
        T = transitions_from([y for m in train for y in data[m][1]])
        Xte, yte = data[mid][0], data[mid][1]
        res["baseline"].append(score(yte, [baseline(x) for x in Xte]))
        P = [model.predict_proba(x) for x in Xte]
        probas[mid] = P
        res["clasificador"].append(score(yte, [majority(np.argmax(p, 1).astype(np.int8), 25) for p in P]))
        for t in taus:
            res[f"clf+markov tau={t}"].append(score(yte, [hmm_decode(p, prior, T, t) for p in P]))
        print(f"  {mid}: baseline {res['baseline'][-1][0]:.3f}  clf {res['clasificador'][-1][0]:.3f}  "
              + "  ".join(f"t{t}={res[f'clf+markov tau={t}'][-1][0]:.3f}" for t in taus) + f"  ({time.time() - t0:.0f}s)",
              flush=True)
    print("\nmétodo                     spellF1  macroF1  posesiones(pred/real)")
    summary = {}
    for k, v in res.items():
        a = np.array(v)
        summary[k] = {"spellF1": float(a[:, 0].mean()), "macroF1": float(a[:, 1].mean())}
        print(f"  {k:24} {a[:, 0].mean():.3f}    {a[:, 1].mean():.3f}    {a[:, 2].mean():.0f}/{a[:, 3].mean():.0f}"
              f"   (por partido: {' '.join(f'{x:.2f}' for x in a[:, 0])})")
    json.dump(summary, open(os.path.join(CACHE, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
