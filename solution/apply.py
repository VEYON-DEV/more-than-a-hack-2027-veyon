"""Paso final: modelo entrenado en IDSSE -> H / A / D para Barça-Athletic.

  1. entrena el gradient boosting con los 7 partidos de IDSSE (tracking ensuciado como el nuestro,
     etiquetas de 'posesión controlada') y mide las transiciones del modelo de Markov ahí;
  2. calcula las mismas pistas (solution/features.py) con nuestro tracking (equipos verificados
     por color si existe data/derived/solution/team_fix.npz) y nuestra pelota (ball_track.npz);
  3. Viterbi por tiempo, con dos reglas seguras del Smart Tagging (automático, no humano):
     de un gol al saque del medio la pelota está muerta, y antes del saque inicial de cada tiempo.

    .venv/bin/python solution/apply.py --out evaluation/results/<equipo>.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (WORK, N_FRAMES, FPS, L_M, W_M, halves, load_atd, team_of_each_track,  # noqa: E402
                    goalkeeper_slots, load_smart_tagging)
from features import features, noisy_players  # noqa: E402
import features as featmod  # noqa: E402
from bench import match_features, make_model, transitions_from, hmm_decode, controlled  # noqa: E402
from switch import change_feats, near_transition, make_det, decode_tv  # noqa: E402
from experiments import add_long, min_duration, stack_feats  # noqa: E402

LABELS = np.array(["H", "A", "D"])


def safe_dead_mask():
    """Pelota muerta con seguridad:
      * goles (Smart Tagging): desde 3 s después del inicio del tag hasta su fin (el saque del
        medio). Los 3 primeros segundos los decide el modelo: el tag puede empezar durante el
        remate, antes de que la pelota entre.
      * antes del saque inicial de cada tiempo: el saque es la marca de games.py (repositorio
        oficial); FAQ.md: el 2.º tiempo se califica desde 110,808 y es todo D hasta el saque."""
    from games import GAMES, GAME, anchor_video_frame
    dead = np.zeros(N_FRAMES, bool)
    for e in load_smart_tagging():
        if e["code"] == "GOALS":
            dead[max(0, e["start"] + 3 * FPS):e["end"]] = True
    g = GAMES[GAME]
    for key in ("kickoff", "kickoff_2nd"):
        first_live = anchor_video_frame(g[key]) + 1
        for _, f0, f1 in halves():
            if f0 + FPS <= first_live <= f1:              # solo si hay un tramo calificado antes del saque
                dead[f0:first_live] = True
    return dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("evaluation", "results", "solucion.csv"))
    ap.add_argument("--tau", type=float, default=0.3, help="temperatura del Markov (elegida en el banco de IDSSE)")
    ap.add_argument("--alpha", type=float, default=2.0, help="escala de cambios H<->A (banco de IDSSE)")
    ap.add_argument("--beta", type=float, default=2.0, help="escala de cambios vivo<->muerto (banco de IDSSE)")
    ap.add_argument("--mindur", type=float, default=3.0, help="duración mínima de posesión en s (banco de IDSSE)")
    ap.add_argument("--no-long", action="store_true", help="sin pistas de ventanas largas")
    ap.add_argument("--no-mindur", action="store_true", help="sin duración mínima de posesión")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1], help="semillas de ensuciado para entrenar")
    ap.add_argument("--tau-train", type=float, default=3.0, help="control mínimo (s) de las etiquetas de entrenamiento")
    ap.add_argument("--g-sw", type=float, default=0.0, help="peso del detector de cambios H<->A (0 = sin detector)")
    ap.add_argument("--g-rs", type=float, default=0.0, help="peso del detector de saques (0 = sin detector)")
    ap.add_argument("--export-feats", default=None, help="guardar las pistas de Barça-Athletic por tiempo (para la red temporal)")
    ap.add_argument("--tcn-probs", default=None, help="npz con P(H,A,D) de la red temporal por tiempo (P0, P1)")
    ap.add_argument("--tcn-weight", type=float, default=0.5, help="peso de la red temporal en la mezcla")
    ap.add_argument("--test-noise", type=float, default=0.0,
                    help="ruido suave (m) sumado a los jugadores de Barça-Athletic al predecir, como el de entrenamiento")
    ap.add_argument("--dump", default=None, help="guardar entradas del decodificador (prior, T, P del GB, detectores, muertos) para pruebas rápidas")
    ap.add_argument("--mirror", action="store_true",
                    help="espejo de equipos: entrenar también con local<->visitante intercambiados y promediar ambas vistas al predecir")
    a = ap.parse_args()

    # 1. entrenamiento en IDSSE
    Xs, ys, ys_seed1 = [], [], []
    for seed in a.seeds:
        for p in sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz"))):
            X2, _, r2 = match_features(p, seed=seed)
            y2 = [controlled(r, int(a.tau_train * FPS)) for r in r2]
            names_train = list(featmod.NAMES)
            if not a.no_long:
                X2 = [add_long(x, names_train) for x in X2]
            Xs += X2
            ys += y2
            if seed == a.seeds[0]:
                ys_seed1 += y2
    Xs_c, ys_c = list(Xs), list(ys)              # el detector de cambios se entrena sin espejo
    if a.mirror:
        for seed in a.seeds:
            for p in sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz"))):
                X2, _, r2 = match_features(p, seed=seed, swap=True)
                if not a.no_long:
                    X2 = [add_long(x, names_train) for x in X2]
                Xs += X2
                ys += [controlled(r, int(a.tau_train * FPS)) for r in r2]
                if seed == a.seeds[0]:
                    ys_seed1 += [controlled(r, int(a.tau_train * FPS)) for r in r2]
    step = 2 if len(a.seeds) == 1 else 3
    Xtr = np.vstack([x[::step] for x in Xs])
    ytr = np.concatenate([y[::step] for y in ys])
    model = make_model().fit(Xtr, ytr)
    prior = np.bincount(ytr, minlength=3) / len(ytr)
    T = transitions_from(ys_seed1)
    tau = a.tau
    use_det = a.g_sw > 0 or a.g_rs > 0
    dets = {}
    if use_det:
        rng = np.random.default_rng(0)
        for kind in ("sw", "rs"):
            Xd, yd = [], []
            for x, y in zip(Xs_c, ys_c):
                t = near_transition(y, kind)
                keep = (t == 1) | (rng.random(len(t)) < 0.25)
                Xd.append(change_feats(x, names_train)[keep])
                yd.append(t[keep])
            dets[kind] = make_det().fit(np.vstack(Xd), np.concatenate(yd))
        print("detectores de cambio entrenados")
    print(f"entrenado con {len(ytr)} frames de IDSSE (semillas {a.seeds}, ventanas largas: {not a.no_long}); "
          f"Markov tau={tau}; duración mínima: {not a.no_mindur}")

    # 2. nuestro partido
    players, _ = load_atd()
    teams = team_of_each_track().astype(np.int8)
    fix_p = os.path.join(WORK, "team_fix.npz")
    team_frames = None
    if os.path.exists(fix_p):
        team_frames = np.load(fix_p)["team"]                 # (N, 46) equipo verificado por color
        print("usando equipos verificados por color")
    gk = np.zeros(46, bool)
    gk[goalkeeper_slots()] = True
    bt = np.load(os.path.join(WORK, "ball_track.npz"))
    dead = safe_dead_mask()
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    exported = {}
    tcnp = np.load(a.tcn_probs) if a.tcn_probs else None
    dump = {"prior": prior, "T": T}
    for hi, (_, f0, f1) in enumerate(halves()):
        pm = players[f0:f1 + 1].astype(np.float64) * [L_M, W_M]
        if a.test_noise > 0:
            pm = noisy_players(pm, a.test_noise, seed=100 + hi)
        if team_frames is not None:
            tf = team_frames[f0:f1 + 1]
            # dos copias de los slots: una con el equipo 0 y otra con el 1, cada frame en la suya
            pm2 = np.concatenate([np.where((tf == 0)[..., None], pm, np.nan),
                                  np.where((tf == 1)[..., None], pm, np.nan)], axis=1)
            t2 = np.r_[np.zeros(46, np.int8), np.ones(46, np.int8)]
            X = features(pm2, t2, np.r_[gk, gk], bt["pos"][f0:f1 + 1], bt["vel"][f0:f1 + 1], L_M, W_M)
            Xw = features(pm2, (1 - t2).astype(np.int8), np.r_[gk, gk], bt["pos"][f0:f1 + 1], bt["vel"][f0:f1 + 1],
                          L_M, W_M) if a.mirror else None
        else:
            X = features(pm, teams, gk, bt["pos"][f0:f1 + 1], bt["vel"][f0:f1 + 1], L_M, W_M)
            Xw = features(pm, (1 - teams).astype(np.int8), gk, bt["pos"][f0:f1 + 1], bt["vel"][f0:f1 + 1],
                          L_M, W_M) if a.mirror else None
        assert list(featmod.NAMES) == names_train
        exported[f"X{hi}"] = X
        if Xw is not None:
            exported[f"Xw{hi}"] = Xw
        if not a.no_long:
            X = add_long(X, names_train)
            if Xw is not None:
                Xw = add_long(Xw, names_train)
        P = model.predict_proba(X)
        if Xw is not None:                       # vista espejada, devuelta a su lugar (H<->A)
            P = 0.5 * (P + model.predict_proba(Xw)[:, [1, 0, 2]])
        dump[f"Pgb{hi}"] = P.copy()
        if tcnp is not None:
            Pt = tcnp[f"P{hi}"][:len(P)]
            P = (1 - a.tcn_weight) * P + a.tcn_weight * Pt
        d = dead[f0:f1 + 1]
        P[d] = [0.001, 0.001, 0.998]
        if use_det:
            Xc = change_feats(X, names_train)
            psw, prs = dets["sw"].predict_proba(Xc)[:, 1], dets["rs"].predict_proba(Xc)[:, 1]
            dump[f"psw{hi}"], dump[f"prs{hi}"], dump[f"dead{hi}"] = psw, prs, d
            path = decode_tv(P, prior, T, tau, a.alpha, a.beta, psw, prs, a.g_sw, a.g_rs)
        else:
            path = hmm_decode(P, prior, T, tau, a.alpha, a.beta)
        if not a.no_mindur and a.mindur > 0:
            path = min_duration(path, int(a.mindur * FPS))
        labels[f0:f1 + 1] = LABELS[path]

    if a.dump:
        np.savez_compressed(a.dump, **dump)
    if a.export_feats:
        np.savez_compressed(a.export_feats, **exported)
        print("pistas exportadas a", a.export_feats)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    g = labels != "X"
    lab = labels[g]
    n_sp, durs = 0, []
    for _, f0, f1 in halves():
        seg = labels[f0:f1 + 1]
        cuts = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(seg)]):
            if seg[s] in "HA":
                durs.append((e - s) / FPS)
    print(f"escrito {a.out}: reparto " + "  ".join(f"{k} {100 * (lab == k).mean():.1f}%" for k in "HAD")
          + f" | {len(durs)} posesiones, media {np.mean(durs):.1f} s")


if __name__ == "__main__":
    main()
