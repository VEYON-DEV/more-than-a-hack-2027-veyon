"""Bancos de diagnóstico: mismo ensuciado que bench.py (DEGRADE del entorno) pero dejando limpia una
parte, para medir cuánto del error viene de la pelota y cuánto de los jugadores.

    DEGRADE=v3rjf .venv/bin/python solution/paradas_build.py --variant clean      -> bench_fclean      (todo limpio)
    DEGRADE=v3rjf .venv/bin/python solution/paradas_build.py --variant ballclean  -> bench_fballclean  (pelota perfecta, jugadores ensuciados)
    DEGRADE=v3rjf .venv/bin/python solution/paradas_build.py --variant playclean  -> bench_fplayclean  (jugadores perfectos, pelota ensuciada)
    DEGRADE=v3rjf .venv/bin/python solution/paradas_build.py --variant ballgaps   -> bench_fballgaps   (pelota exacta pero con los mismos huecos; jugadores ensuciados)
Mismo formato de archivo que el banco (X0, X1, y0, y1, raw0, raw1, names).
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, FPS  # noqa: E402


def carry_ball(p, b, bc, lab, rng, pc, pitch=None, hold=True, p_ret=0.0):
    """Pelota "en los pies" durante las paradas, como en Barça-Athletic. Medido sin etiquetas:
      * en los tramos en que todos caminan, nuestro tracking ve la pelota pegada a un jugador (el que la lleva, la
        devuelve o la acomoda) el 9.6 % del tiempo contra 2 % en IDSSE ensuciado;
      * en los 10 s previos a los saques que marca el Smart Tagging la pelota está a < 1.5 m de un jugador el 20 %
        del tiempo (39 % antes de un saque de arco, con tramos de >= 3 s en el 40 %) contra 7 % (4 %) en IDSSE.
    En una fracción pc de las paradas se agregan 1-3 episodios de 0.6-3 s en que la pelota se ve junto a un jugador
    (siguiéndolo) o quieta; la mitad de las veces el primero termina justo en la reanudación (el que saca). Además
    (hold): en el 70 % de los saques de arco y el 5 % del resto, el que saca tiene la pelota (o está quieta en el
    punto del saque) durante los últimos 2.5-7 s."""
    b = b.copy()
    n = len(lab)
    e = np.flatnonzero(np.diff(np.r_[0, (lab == 2).astype(np.int8), 0]))

    def nearest(t, ref, dmax):
        d = np.hypot(p[t, :, 0] - ref[0], p[t, :, 1] - ref[1])
        if np.all(np.isnan(d)) or np.nanmin(d) > dmax:
            return None
        return int(np.nanargmin(d))

    def put(a0, a1, k, tref, follow):
        if follow:
            ang = rng.uniform(0, 2 * np.pi)
            b[a0:a1] = p[a0:a1, k] + 0.35 * np.array([np.cos(ang), np.sin(ang)]) + rng.normal(0, 0.08, (a1 - a0, 2))
        else:
            b[a0:a1] = p[tref, k] + rng.normal(0, 0.08, (a1 - a0, 2))

    for s0, e0 in zip(e[::2], e[1::2]):
        if e0 >= n - 1 or e0 - s0 < 3 * FPS:
            continue
        spot = bc[e0] if not np.isnan(bc[e0, 0]) else bc[min(n - 1, e0 + 5)]
        if rng.random() < pc:
            for j in range(1 + int(rng.poisson(0.8))):
                L = int(rng.uniform(0.6, 3.0) * FPS)
                if j == 0 and rng.random() < 0.5:
                    a0, a1, ref, tref = max(s0 + 12, e0 - L), e0, spot, e0
                else:
                    if e0 - s0 - L - 25 <= 0:
                        continue
                    a0 = int(rng.integers(s0 + 12, e0 - L - 12))
                    a1, ref, tref = a0 + L, bc[a0], a0
                if np.isnan(ref[0]):
                    continue
                k = nearest(tref, ref, 8.0)
                if k is not None:
                    put(a0, a1, k, tref, rng.random() < 0.7)
        if p_ret > 0 and e0 - s0 >= 8 * FPS and not np.isnan(spot[0]) and rng.random() < p_ret:
            # "devolución": durante la parada un jugador le pasa la pelota al que va a sacar (10-35 m, 6-12 m/s),
            # que la tiene 0.8-3 s hasta la reanudación. En Barça-Athletic la pelota conocida se mueve a > 2 m/s
            # el 23 % del tiempo en que todos caminan, contra 6-9 % en IDSSE ensuciado (v3).
            k = nearest(e0, spot, 6.0)
            if k is not None:
                t_arr = e0 - int(rng.uniform(0.8, 3) * FPS)
                dist, speed = rng.uniform(10, 35), rng.uniform(6, 12)
                t_dep = t_arr - max(5, int(dist / speed * FPS))
                if t_dep > s0 + 25 and not np.isnan(p[t_arr, k, 0]):
                    dest = p[t_arr, k]
                    d = np.hypot(p[t_dep, :, 0] - dest[0], p[t_dep, :, 1] - dest[1])
                    d[k] = np.nan
                    if not np.all(np.isnan(d)):
                        j = int(np.nanargmin(np.abs(d - dist)))
                        orig = p[t_dep, j]
                        a0 = max(s0 + 12, t_dep - int(rng.uniform(0.5, 2.0) * FPS))
                        put(a0, t_dep, j, t_dep, True)
                        u = 1 - (1 - np.linspace(0, 1, t_arr - t_dep, endpoint=False)) ** 1.5
                        b[t_dep:t_arr] = orig + u[:, None] * (dest - orig) + rng.normal(0, 0.05, (t_arr - t_dep, 2))
                        put(t_arr, e0, k, e0, True)
                        continue
        if hold and not np.isnan(spot[0]):
            gk_like = pitch is not None and (spot[0] < 7 or spot[0] > pitch[0] - 7) and abs(spot[1] - pitch[1] / 2) < 11
            if rng.random() < (0.7 if gk_like else 0.05):
                k = nearest(e0, spot, 6.0)
                if k is not None:
                    L = int(rng.uniform(2.5, 7) * FPS)
                    a0 = max(s0 + 12, e0 - L)
                    if rng.random() < 0.7:
                        put(a0, e0, k, e0, True)
                    else:
                        b[a0:e0] = spot + rng.normal(0, 0.08, (e0 - a0, 2))
    return b


class _V34(str):
    """Ensuciado v3 (jugadores suavizados) + los extras de v4 (pelota fuera de las líneas casi nunca visible, menos
    jugadores sueltos perdidos, varios perdidos a la vez). En bench.py, DEGRADE=v4 NO aplica el suavizado de v3
    (la condición es startswith("v3")), así que el banco v4 tiene jugadores sin suavizar; esto los combina."""
    def startswith(self, prefix, *a):
        return prefix in ("v3", "v4")


def job(args):
    import bench
    import ball_track
    import features as featmod
    from features import features
    path, seed, swap, variant, out_dir, sig, siga, p_carry, p_ret, v34 = args
    if v34:
        bench.DEGRADE = _V34("v3")
    if siga:                                   # ruido de proceso del Kalman (m/s²); por defecto ball_track.SIGMA_A = 20
        DT = ball_track.DT
        q1 = np.array([[DT ** 4 / 4, DT ** 3 / 2], [DT ** 3 / 2, DT ** 2]]) * siga ** 2
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q1
        Q[np.ix_([1, 3], [1, 3])] = q1
        ball_track.Q = Q
    z = np.load(path)
    rng = np.random.default_rng(seed)
    L, W = z["pitch"]
    Xs, ys, raws = [], [], []
    for h in (0, 1):
        pc, bc = z[f"players_{h}"].astype(np.float64), z[f"ball_{h}"].astype(np.float64)
        live = bench.controlled(z[f"label_{h}"], int(1.5 * FPS)) != 2
        p, b, flip = bench.degrade(z[f"players_{h}"], z[f"ball_{h}"], rng, live=live, pitch=(L, W), teams=z["teams"],
                                   gk=z["gk"])
        if variant == "clean":
            p, b, flip = pc, bc, np.zeros(pc.shape[:2], bool)
        elif variant == "ballclean":
            b = bc
        elif variant == "playclean":
            p, flip = pc, np.zeros(pc.shape[:2], bool)
        elif variant == "ballgaps":
            b = np.where(np.isnan(b), np.nan, bc)
        elif variant == "carry":
            b = carry_ball(p, b, bc, z[f"label_{h}"], rng, p_carry, pitch=(float(L), float(W)), p_ret=p_ret)
        n = len(b)
        xs, _, _ = ball_track.track_half(0, n - 1, {}, b, atd_sigma=np.full(n, sig) if sig else None)
        pos, vel = xs[:, :2], xs[:, 2:]
        teams = z["teams"] if not swap else (1 - z["teams"]).astype(np.int8)
        pf = np.concatenate([np.where(flip[..., None], np.nan, p), np.where(flip[..., None], p, np.nan)], axis=1)
        X = features(pf, np.r_[teams, 1 - teams], np.r_[z["gk"], z["gk"]], pos, vel, L, W)
        Xs.append(X)
        lab_h = bench.swap_labels(z[f"label_{h}"]) if swap else z[f"label_{h}"]
        ys.append(bench.controlled(lab_h, int(bench.TAU_CTRL * FPS)))
        raws.append(lab_h)
    mid = os.path.basename(path)[:-4]
    np.savez_compressed(os.path.join(out_dir, f"{mid}_s{seed}{'w' if swap else ''}.npz"), X0=Xs[0], X1=Xs[1], y0=ys[0],
                        y1=ys[1], raw0=raws[0], raw1=raws[1], names=np.array(featmod.NAMES))
    return mid, seed, swap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["clean", "ballclean", "playclean", "ballgaps", "full", "carry"])
    ap.add_argument("--pc", type=float, default=0.5, help="carry: fracción de paradas con la pelota en los pies antes del saque")
    ap.add_argument("--sig", type=float, default=0.0, help="error (m) que el Kalman supone para la pelota medida (por defecto 0.8)")
    ap.add_argument("--siga", type=float, default=0.0, help="ruido de proceso del Kalman (m/s²; por defecto 20)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--v34", action="store_true", help="ensuciado v3 + extras de v4 (ver _V34)")
    ap.add_argument("--ret", type=float, default=0.0, help="carry: fracción de paradas largas con una devolución (pase al que saca)")
    a = ap.parse_args()
    out_dir = os.path.join(WORK, f"bench_f{a.variant}" + (f"_s{a.sig:g}" if a.sig else "") + (f"_a{a.siga:g}" if a.siga else "")
                           + (f"_p{a.pc:g}" if a.variant == "carry" else "") + a.suffix)
    os.makedirs(out_dir, exist_ok=True)
    seeds = [1] if a.variant == "clean" else a.seeds
    jobs = [(p, s, w, a.variant, out_dir, a.sig, a.siga, a.pc, a.ret, a.v34) for p in sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz")))
            for s in seeds for w in (False, True)]
    t = time.time()
    with Pool(14) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print(f"listo: {len(jobs)} archivos en {out_dir} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
