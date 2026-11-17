"""Etiquetas de "posesión controlada" por contacto, calculadas en IDSSE con el tracking LIMPIO.

La definición oficial (CHALLENGE.md) no es por duración: un toque sin control, un desvío o un duelo aéreo ganado
sin control posterior NO cambian la posesión; y en la verdad pública hay posesiones de menos de 1.5 s. Aquí un tramo
de las etiquetas crudas de la DFL (por toques) cuenta como posesión si el equipo CONTROLA la pelota:
  * algún jugador suyo la mantiene a menos de RC metros (y a menos de Z_KEEP de altura) T_KEEP segundos seguidos, o
  * hay dos contactos separados del mismo equipo (un pase que llega a un compañero), o
  * es el tramo que sigue a una reanudación (tras una parada hay un poseedor claro).
Si no, la posesión sigue con el equipo anterior (como en bench.controlled, pero por control y no por duración).

    .venv/bin/python solution/paradas_ballz.py           # altura de la pelota (una vez)
    .venv/bin/python solution/paradas_ctrl.py            # estadísticas y data/derived/solution/paradas/ctrl_labels.npz
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, FPS  # noqa: E402

RC = float(os.environ.get("CTRL_RC", "1.2"))          # m: distancia jugador-pelota (en el piso) que cuenta como contacto
T_KEEP = float(os.environ.get("CTRL_KEEP", "0.5"))    # s seguidos de contacto = "la mantiene"
Z_KEEP = 1.5                                          # m: para "mantenerla" la pelota tiene que estar baja
Z_TOUCH = 2.6                                         # m: por encima de esto nadie la toca (pasa por arriba)
FAR = 2.5                                             # m: la pelota se separó del equipo (entre dos contactos)
LEAD = int(0.4 * FPS)                                 # el contacto puede empezar un poco antes del cambio de etiqueta


def team_dist(players, teams, ball):
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    return d[:, teams == 0].min(1), d[:, teams == 1].min(1)


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def merge_runs(mask, gap=3):
    out = []
    for r in runs(mask):
        if out and r[0] - out[-1][1] <= gap:
            out[-1] = (out[-1][0], r[1])
        else:
            out.append(r)
    return out


def control_evidence(dX, z, s, e):
    """(segundos del contacto bajo seguido más largo, número de contactos separados) del equipo en [s-LEAD, e)."""
    a = max(0, s - LEAD)
    seg, zz = dX[a:e], z[a:e]
    keep = max((b - a_ for a_, b in merge_runs((seg <= RC) & (zz <= Z_KEEP))), default=0) / FPS
    n_ep, last_end = 0, None
    for a_, b in merge_runs((seg <= RC) & (zz <= Z_TOUCH)):
        if last_end is None or seg[last_end:a_].max(initial=0) > FAR:
            n_ep += 1
        last_end = b
    return keep, n_ep


def ctrl_labels(players, teams, ball, z, raw, keep_s=None):
    keep_s = T_KEEP if keep_s is None else keep_s
    dH, dA = team_dist(players.astype(np.float64), teams, ball.astype(np.float64))
    z = np.nan_to_num(z.astype(np.float64), nan=0.0)
    lab = raw.copy()
    cuts = np.flatnonzero(raw[1:] != raw[:-1]) + 1
    st, en = np.r_[0, cuts], np.r_[cuts, len(raw)]
    cur = 2                                       # equipo con la posesión vigente (2 = pelota muerta)
    info = []
    for s, e in zip(st, en):
        x = raw[s]
        if x == 2:
            cur = 2
            continue
        keep, n_ep = control_evidence(dH if x == 0 else dA, z, s, e)
        ok = (cur == 2) or (x == cur) or keep >= keep_s or n_ep >= 2
        info.append(((e - s) / FPS, keep, n_ep, bool(ok), cur == 2))
        if ok:
            cur = x
        else:
            lab[s:e] = cur
    return lab, info


def spell_stats(labs, name):
    sp, tot, nD = [], 0, 0
    for w in labs:
        cuts = np.flatnonzero(w[1:] != w[:-1]) + 1
        for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(w)]):
            if w[s] < 2:
                sp.append((e - s) / FPS)
        tot += len(w)
        nD += int((w == 2).sum())
    sp = np.array(sp)
    lm = (tot - nD) / FPS / 60
    print(f"  {name:22} posesiones {len(sp):5d} ({len(sp) / lm:4.2f}/min vivo) media {sp.mean():5.1f}s mediana {np.median(sp):4.1f}s "
          f"<1.5s {100 * (sp < 1.5).mean():4.1f}%  <3s {100 * (sp < 3).mean():4.1f}%  <5s {100 * (sp < 5).mean():4.1f}%")


def main():
    global RC
    from bench import controlled
    zz = np.load(os.path.join(WORK, "paradas", "idsse_ballz.npz"))
    out = {}
    allv = {k: [] for k in ("cruda", "control>=1.5s", "control>=3s", "CONTROL por contacto")}
    infos = []
    for p in sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz"))):
        z = np.load(p)
        mid = os.path.basename(p)[:-4]
        for h in (0, 1):
            raw = z[f"label_{h}"]
            lab, info = ctrl_labels(z[f"players_{h}"], z["teams"], z[f"ball_{h}"], zz[f"{mid}_z{h}"], raw)
            infos += info
            out[f"{mid}_c{h}"] = lab
            allv["cruda"].append(raw)
            allv["control>=1.5s"].append(controlled(raw, 37))
            allv["control>=3s"].append(controlled(raw, 75))
            allv["CONTROL por contacto"].append(lab)
    print(f"definiciones en IDSSE (7 partidos), RC={RC} m, T_KEEP={T_KEEP} s:")
    for k, v in allv.items():
        spell_stats(v, k)
    I = np.array([(a, b, c, d, e) for a, b, c, d, e in infos], float)
    nr = I[I[:, 4] == 0]                       # tramos crudos que no siguen a una parada
    print(f"tramos crudos (sin contar los que siguen a una parada): {len(nr)}; con control {100 * nr[:, 3].mean():.1f}%")
    for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 3), (3, 5), (5, 1e9)):
        s = nr[(nr[:, 0] >= lo) & (nr[:, 0] < hi)]
        if len(s):
            print(f"    duración {lo}-{hi if hi < 1e8 else 'inf'} s: n={len(s):5d}  con control {100 * s[:, 3].mean():5.1f}%  "
                  f"(mantiene>={T_KEEP}s: {100 * (s[:, 1] >= T_KEEP).mean():5.1f}%, dos contactos: {100 * (s[:, 2] >= 2).mean():5.1f}%, "
                  f"ningún contacto: {100 * (s[:, 2] == 0).mean():5.1f}%)")
    # variantes de los umbrales (para medir cuánto depende el resultado de ellos): 2 = estricta, 3 = laxa
    for suf, rc, keep in (("2", 1.0, 0.8), ("3", 1.5, 0.3)):
        RC = rc
        v = []
        for p in sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz"))):
            z = np.load(p)
            mid = os.path.basename(p)[:-4]
            for h in (0, 1):
                out[f"{mid}_c{suf}{h}"] = ctrl_labels(z[f"players_{h}"], z["teams"], z[f"ball_{h}"], zz[f"{mid}_z{h}"],
                                                      z[f"label_{h}"], keep_s=keep)[0]
                v.append(out[f"{mid}_c{suf}{h}"])
        spell_stats(v, f"variante {suf} (RC={rc}, {keep}s)")
    np.savez_compressed(os.path.join(WORK, "paradas", "ctrl_labels.npz"), **out)
    # objetivos auxiliares "privilegiados" (solo existen con el tracking limpio de IDSSE): por frame, si un jugador
    # local / visitante está en contacto con la pelota (cerca y pelota baja) y si la pelota va por el aire
    RC = float(os.environ.get("CTRL_RC", "1.2"))
    aux = {}
    for p in sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz"))):
        z = np.load(p)
        mid = os.path.basename(p)[:-4]
        for h in (0, 1):
            dH, dA = team_dist(z[f"players_{h}"].astype(np.float64), z["teams"], z[f"ball_{h}"].astype(np.float64))
            zb = np.nan_to_num(zz[f"{mid}_z{h}"].astype(np.float64), nan=0.0)
            live = z[f"label_{h}"] != 2
            aux[f"{mid}_k{h}"] = np.stack([(dH <= RC) & (zb <= Z_TOUCH) & live, (dA <= RC) & (zb <= Z_TOUCH) & live,
                                           (zb > Z_KEEP) & live], 1).astype(np.uint8)
    np.savez_compressed(os.path.join(WORK, "paradas", "aux_targets.npz"), **aux)
    a = np.concatenate(list(aux.values()))
    print("objetivos auxiliares: contacto local %.3f, contacto visitante %.3f, pelota en el aire %.3f de los frames" % tuple(a.mean(0)))


if __name__ == "__main__":
    main()
