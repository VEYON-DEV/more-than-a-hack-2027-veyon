"""Aplica las redes multi-definición (paradas_tcn.py --only-all) a Barça-Athletic y escribe el CSV.

  1. .venv-torch/bin/python solution/paradas_apply.py predict --tags mc_m1 mc_m2 mc_m3 --out data/derived/solution/paradas/bar_mc.npz
        promedia las redes (vista normal y espejada) -> P0, P1 (n, K, 3), S0, S1 (n, K), taus
  2. .venv/bin/python solution/paradas_apply.py csv --probs data/derived/solution/paradas/bar_mc.npz --head -1 --out <archivo.csv>
        Viterbi de la entrega (temperatura 0.4, alpha 2, beta 0.6, peso del detector 1.5) con la cabeza elegida,
        transiciones medidas en IDSSE con esa definición, duración mínima y las dos reglas seguras de apply.py.
     También guarda <archivo>.npz con P0, P1 (n, 3) de esa cabeza, en el formato de bar_tcn_*.npz.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "data", "derived", "solution")
OUT = os.path.join(WORK, "paradas")
STRIDE = 2


def predict(a):
    import torch
    from paradas_tcn import Net, prep
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    z = np.load(a.feats)
    acc = {}
    taus = None
    for tag in a.tags:
        ck = torch.load(os.path.join(a.models_dir, f"{tag}.pt"), map_location=dev, weights_only=False)
        g = ck["args"]
        taus = g["taus"]
        model = Net(2 * len(ck["mu"]), len(taus), c=g["c"], dil=tuple(g["dil"]), drop=g["drop"], stages=g["stages"],
                    naux=3 if g.get("aux", 0) > 0 else 0).to(dev)
        model.load_state_dict(ck["state"])
        model.eval()
        mu, sd = torch.tensor(ck["mu"], device=dev), torch.tensor(ck["sd"], device=dev)
        K = len(taus)
        for h in (0, 1):
            ps, ss = [], []
            for key, perm in ((f"X{h}", [0, 1, 2]), (f"Xw{h}", [1, 0, 2])):
                with torch.no_grad():
                    x = prep(torch.tensor(z[key][::STRIDE].astype(np.float32), device=dev)[None], mu, sd).transpose(1, 2)
                    st, lsw, _ = model(x)
                    p = torch.softmax(st[-1][0].reshape(K, 3, -1), 1).permute(2, 0, 1).cpu().numpy()[:, :, perm]
                    s = torch.sigmoid(lsw[0]).T.cpu().numpy()
                ps.append(p)
                ss.append(s)
            n = len(z[f"X{h}"])
            acc[f"P{h}"] = acc.get(f"P{h}", 0) + np.repeat(0.5 * (ps[0] + ps[1]), STRIDE, 0)[:n] / len(a.tags)
            acc[f"S{h}"] = acc.get(f"S{h}", 0) + np.repeat(0.5 * (ss[0] + ss[1]), STRIDE, 0)[:n] / len(a.tags)
    np.savez_compressed(a.out, taus=np.array(taus), **{k: v.astype(np.float32) for k, v in acc.items()})
    print("probabilidades ->", a.out, "| redes:", len(a.tags), "| definiciones:", taus)


def csv(a):
    import paradas_lib as fl
    from common import N_FRAMES, FPS, halves
    from bench import controlled, transitions_from, swap_labels
    from switch import decode_tv
    from apply import safe_dead_mask
    z = np.load(a.probs)
    taus = [float(t) for t in z["taus"]]
    name = {"c": -1.0, "c2": -2.0, "c3": -3.0}
    mix = {(name[k_] if k_ in name else float(k_)): float(v) for k_, v in (it.split(":") for it in a.mix.split(","))} \
        if a.mix else {a.head: 1.0}
    wk = np.array([mix.get(t, 0.0) for t in taus])
    wk = wk / wk.sum()
    ys = [y for m in fl.MIDS for y in fl.truth(m, a.head)]
    ys = ys + [swap_labels(y) for y in ys]
    prior = np.bincount(np.concatenate([y[::3] for y in ys]), minlength=3).astype(float)
    prior /= prior.sum()
    T = transitions_from(ys)
    dead = safe_dead_mask()
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    md = max(a.head, 0.0) if a.mindur is None else a.mindur
    save = {}
    for hi, (_, f0, f1) in enumerate(halves()):
        P = (z[f"P{hi}"].astype(np.float64) * wk[None, :, None]).sum(1)
        S = (z[f"S{hi}"].astype(np.float64) * wk[None, :]).sum(1)
        P[dead[f0:f1 + 1]] = [0.001, 0.001, 0.998]
        save[f"P{hi}"] = P.astype(np.float32)
        path = decode_tv(P, prior, T, a.cfg[0], a.cfg[1], a.cfg[2], S, np.ones(len(P)), a.cfg[3], 0.0)
        if md > 0:
            path = controlled(path, int(round(md * FPS)))
        labels[f0:f1 + 1] = np.array(["H", "A", "D"])[path]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    np.savez_compressed(os.path.splitext(a.out)[0] + ".npz", **save)
    g = labels != "X"
    durs = []
    for _, f0, f1 in halves():
        seg = labels[f0:f1 + 1]
        cuts = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        durs += [(e - s) / FPS for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(seg)]) if seg[s] in "HA"]
    durs = np.array(durs)
    print(f"escrito {a.out}: " + "  ".join(f"{c} {100 * (labels[g] == c).mean():.1f}%" for c in "HAD")
          + f" | {len(durs)} posesiones, media {durs.mean():.1f} s, mediana {np.median(durs):.1f} s, <1.5 s {100 * (durs < 1.5).mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("predict")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--feats", default=os.path.join(WORK, "bar_feats.npz"))
    p.add_argument("--models-dir", default=os.path.join(OUT, "models"),
                   help="carpeta con los pesos <tag>.pt (la entrega usa solution/models/paradas)")
    p.add_argument("--out", required=True)
    c = sp.add_parser("csv")
    c.add_argument("--probs", required=True)
    c.add_argument("--head", type=float, required=True)
    c.add_argument("--mix", default=None, help='mezcla de cabezas "1.5:1,c:1" (c = control por contacto); --head da las transiciones')
    c.add_argument("--mindur", type=float, default=None)
    c.add_argument("--cfg", type=float, nargs=4, default=[0.4, 2.0, 0.6, 1.5])
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    (predict if a.cmd == "predict" else csv)(a)


if __name__ == "__main__":
    main()
