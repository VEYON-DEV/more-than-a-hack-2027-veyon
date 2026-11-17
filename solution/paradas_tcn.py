"""Red temporal multi-definición: una sola red predice la posesión con varias definiciones de
"control" a la vez (etiquetas crudas de la DFL por toques, y control mínimo de 0.5 / 1.0 / 1.5 / 3.0 s) y,
para cada una, la probabilidad de que haya un cambio H<->A (reemplaza al detector de gradient boosting).

Mismas pistas por frame que tcn.py (77 + máscara de NaN), mismo esqueleto (convoluciones dilatadas), pero:
  * varias cabezas (una por definición) + cabezas de cambio (sw por definición, rs = saque, ds = pelota muerta)
  * pesos promediados durante el entrenamiento (EMA), que bajan la variación entre semillas
  * datos y lotes en la GPU (un partido afuera cada vez en pocos minutos)

    .venv-torch/bin/python solution/paradas_tcn.py --tag md1 --train bench_v3rjf --test bench_v3rjf bench_v3 --model-seeds 1 2 3
Salida: data/derived/solution/paradas/<tag>_m<semilla>__<banco de prueba>.npz
          <partido>_P<h>  (n, K, 3) probabilidades H/A/D por definición (espejo promediado)
          <partido>_S<h>  (n, K)    probabilidad de cambio H<->A por definición
        data/derived/solution/paradas/models/<tag>_m<semilla>.pt  (con --all: red entrenada con los 7 partidos)
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "data", "derived", "solution")
OUT = os.path.join(WORK, "paradas")
FPS, STRIDE = 25, 2
AUX_ON = False


def controlled(lab, tau_f):
    """Igual que bench.controlled: un tramo de un equipo más corto que tau_f no cuenta."""
    lab = lab.copy()
    if tau_f <= 1:
        return lab
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


def near(y, kind, tol):
    t = np.flatnonzero(y[1:] != y[:-1]) + 1
    a, b = y[t - 1], y[t]
    sel = {"sw": (a != 2) & (b != 2), "rs": (a == 2) & (b != 2), "ds": (a != 2) & (b == 2)}[kind]
    out = np.zeros(len(y), np.float32)
    for tt in t[sel]:
        out[max(0, tt - tol):tt + tol + 1] = 1
    return out


def spell_weights(y, ref, gamma, wmax):
    """Peso por frame ~ (ref / duración del tramo)^gamma: cada posesión corta pesa más (la métrica cuenta
    posesiones, no frames). Los frames de pelota muerta pesan 1."""
    w = np.ones(len(y), np.float32)
    if gamma <= 0:
        return w
    cuts = np.flatnonzero(y[1:] != y[:-1]) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(y)]):
        if y[s] != 2:
            w[s:e] = np.clip((ref / (e - s)) ** gamma, 1.0, wmax)
    return w


class Block(nn.Module):
    def __init__(self, c, d, drop, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=(k // 2) * d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 1)
        self.n = nn.GroupNorm(8, c)
        self.dp = nn.Dropout(drop)

    def forward(self, x):
        return x + self.c2(self.dp(torch.relu(self.n(self.c1(x)))))


class Net(nn.Module):
    def __init__(self, fin, K, c=96, dil=(1, 2, 4, 8, 16, 32), drop=0.1, stages=1, naux=0):
        super().__init__()
        self.K = K
        self.auxh = nn.Conv1d(c, naux, 1) if naux else None      # objetivos privilegiados (contactos, pelota aérea)
        self._aux = None
        self.inp = nn.Conv1d(fin, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d, drop) for d in dil])
        self.out = nn.Conv1d(c, 4 * K + 2, 1)          # por definición: 3 estados + 1 cambio; + rs, ds
        # etapas de refinamiento (MS-TCN): miran las probabilidades de la etapa anterior + las pistas ocultas
        self.ref = nn.ModuleList([nn.ModuleDict({"inp": nn.Conv1d(3 * K + c, c, 1),
                                                 "blocks": nn.Sequential(*[Block(c, d, drop) for d in dil]),
                                                 "out": nn.Conv1d(c, 3 * K, 1)}) for _ in range(stages - 1)])

    def forward(self, x):                              # x: (B, F, T)
        h = self.blocks(torch.relu(self.inp(x)))
        o = self.out(h)
        self._aux = self.auxh(h) if self.auxh is not None else None
        K = self.K
        st = [o[:, :3 * K]]
        for r in self.ref:
            p = torch.softmax(st[-1].reshape(o.shape[0], K, 3, -1), 2).reshape(o.shape[0], 3 * K, -1)
            st.append(r["out"](r["blocks"](torch.relu(r["inp"](torch.cat([p, h], 1))))))
        return st, o[:, 3 * K:4 * K], o[:, 4 * K:]      # estados por etapa, sw por definición, (rs, ds)


_CTRL = None
_AUX = None


def aux_for(m, h, swapped):
    global _AUX
    if _AUX is None:
        _AUX = np.load(os.path.join(OUT, "aux_targets.npz"))
    k = _AUX[f"{m}_k{h}"].astype(np.float32)
    return k[:, [1, 0, 2]] if swapped else k


def labels_for(m, h, raw, swapped, t):
    """t >= 0: control mínimo de t segundos (0 = etiquetas crudas). t < 0: control por contacto (paradas_ctrl.py)."""
    global _CTRL
    if t >= 0:
        return controlled(raw, int(round(t * FPS)))
    if _CTRL is None:
        _CTRL = np.load(os.path.join(OUT, "ctrl_labels.npz"))
    lab = _CTRL[f"{m}_c{'' if t == -1 else int(-t)}{h}"].copy()      # -1 base, -2 estricta, -3 laxa
    if swapped:
        hh, aa = lab == 0, lab == 1
        lab[hh], lab[aa] = 1, 0
    assert len(lab) == len(raw) and np.array_equal(lab == 2, raw == 2)
    return lab


def load_sub(sub, seeds, taus, tol, sw_gamma):
    paths = sorted(glob.glob(os.path.join(WORK, sub, "J03W*_s1.npz")))
    mids = [os.path.basename(p).split("_")[0] for p in paths]
    data = {}
    for s in seeds:
        for m in mids:
            for w in (False, True):
                z = np.load(os.path.join(WORK, sub, f"{m}_s{s}{'w' if w else ''}.npz"))
                for h in (0, 1):
                    raw = z[f"raw{h}"]
                    ys = [labels_for(m, h, raw, w, t)[::STRIDE] for t in taus]
                    data[(sub, s, m, w, h)] = {
                        "X": z[f"X{h}"][::STRIDE].astype(np.float32), "n": len(raw),
                        "y": np.stack(ys, 1).astype(np.int64),
                        "sw": np.stack([near(y, "sw", tol) for y in ys], 1),
                        "rd": np.stack([near(ys[0], "rs", tol), near(ys[0], "ds", tol)], 1),
                        "ax": aux_for(m, h, w)[::STRIDE] if AUX_ON else np.zeros((len(ys[0]), 3), np.float32),
                        "w": np.stack([spell_weights(y, 5 * FPS / STRIDE, sw_gamma, 4.0) for y in ys], 1)}
    return mids, data


def stats(seqs):
    c = sum((~np.isnan(d["X"])).sum(0).astype(np.float64) for d in seqs)
    s1 = sum(np.nansum(d["X"].astype(np.float64), 0) for d in seqs)
    s2 = sum(np.nansum(d["X"].astype(np.float64) ** 2, 0) for d in seqs)
    mu = s1 / np.maximum(c, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(c, 1) - mu ** 2, 0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def prep(x, mu, sd):
    nanm = torch.isnan(x)
    z = torch.where(nanm, torch.zeros_like(x), (x - mu) / sd)
    return torch.cat([z, nanm.float()], -1)


def train(seqs, mu, sd, a, seed, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    K = len(a.taus)
    fin = 2 * seqs[0]["X"].shape[1]
    model = Net(fin, K, c=a.c, dil=tuple(a.dil), drop=a.drop, stages=a.stages, naux=3 if a.aux > 0 else 0).to(dev)
    ema = copy.deepcopy(model) if a.ema > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    lens = np.array([len(d["X"]) for d in seqs])
    pw = lens / lens.sum()
    hw = torch.tensor(a.head_w, device=dev)
    T, B = a.T, a.B
    for it in range(a.steps):
        idx = rng.choice(len(seqs), B, p=pw)
        st = [int(rng.integers(0, max(1, lens[i] - T))) for i in idx]
        xb = prep(torch.stack([seqs[i]["Xg"][s:s + T] for i, s in zip(idx, st)]), mu, sd).transpose(1, 2)
        yb = torch.stack([seqs[i]["yg"][s:s + T] for i, s in zip(idx, st)])          # (B, T, K)
        sb = torch.stack([seqs[i]["swg"][s:s + T] for i, s in zip(idx, st)])         # (B, T, K)
        rb = torch.stack([seqs[i]["rdg"][s:s + T] for i, s in zip(idx, st)])         # (B, T, 2)
        wb = torch.stack([seqs[i]["wg"][s:s + T] for i, s in zip(idx, st)])          # (B, T, K)
        ab = torch.stack([seqs[i]["axg"][s:s + T] for i, s in zip(idx, st)]) if a.aux > 0 else None
        stages, lsw, lrd = model(xb)
        loss = 0.0
        for lo in stages:
            lo = lo.reshape(B, K, 3, -1)                                             # (B, K, 3, T)
            ce = Fn.cross_entropy(lo.permute(0, 2, 3, 1), yb, reduction="none")      # (B, T, K)
            loss = loss + ((ce * wb).sum((0, 1)) / wb.sum((0, 1)) * hw).sum()
            if a.tmse > 0:                                                           # suavidad (MS-TCN)
                lp = torch.log_softmax(lo, 2)
                loss = loss + a.tmse * torch.clamp((lp[..., 1:] - lp[..., :-1].detach()) ** 2, max=16).mean() * K
        if a.bw > 0:
            loss = loss + a.bw * ((Fn.binary_cross_entropy_with_logits(lsw.transpose(1, 2), sb, reduction="none")
                                   .mean((0, 1)) * hw).sum()
                                  + Fn.binary_cross_entropy_with_logits(lrd.transpose(1, 2), rb))
        if a.aux > 0:
            loss = loss + a.aux * Fn.binary_cross_entropy_with_logits(model._aux.transpose(1, 2), ab)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if ema is not None:
            d = min(a.ema, (1 + it) / (10 + it))
            with torch.no_grad():
                for pe, pm_ in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm_.detach(), alpha=1 - d)
    return ema if ema is not None else model


@torch.no_grad()
def predict(model, X, mu, sd, dev):
    model.eval()
    x = prep(torch.tensor(X, device=dev)[None], mu, sd).transpose(1, 2)
    stages, lsw, _ = model(x)
    K = model.K
    p = torch.softmax(stages[-1][0].reshape(K, 3, -1), 1).permute(2, 0, 1).cpu().numpy()   # (T, K, 3)
    s = torch.sigmoid(lsw[0]).T.cpu().numpy()                                               # (T, K)
    return p, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", nargs="+", default=["bench_v3rjf"], help="bancos de entrenamiento")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--test", nargs="+", default=["bench_v3rjf"], help="bancos de prueba (semilla 1)")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 3.0])
    ap.add_argument("--head-w", type=float, nargs="+", default=None)
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--c", type=int, default=96)
    ap.add_argument("--dil", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--tmse", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--bw", type=float, default=1.0, help="peso de las cabezas de cambio")
    ap.add_argument("--tol", type=int, default=2, help="+- muestras (a 12.5 Hz) alrededor de un cambio")
    ap.add_argument("--aux", type=float, default=0.0, help="peso de los objetivos privilegiados (contactos reales, pelota aérea)")
    ap.add_argument("--spell-gamma", type=float, default=0.0, help="peso extra de las posesiones cortas en la pérdida")
    ap.add_argument("--folds", nargs="*", default=None, help="partidos a dejar afuera (por defecto todos)")
    ap.add_argument("--all", action="store_true", help="además, entrenar con los 7 y guardar los pesos")
    ap.add_argument("--only-all", action="store_true")
    a = ap.parse_args()
    a.head_w = a.head_w or [1.0] * len(a.taus)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    global AUX_ON
    AUX_ON = a.aux > 0
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    t0 = time.time()
    data, mids = {}, None
    for sub in a.train:
        mids, d = load_sub(sub, a.train_seeds, a.taus, a.tol, a.spell_gamma)
        data.update(d)
    test = {}
    for sub in a.test:
        if sub in a.train and 1 in a.train_seeds:
            test.update({k: v for k, v in data.items() if k[0] == sub and k[1] == 1})
        else:
            test.update(load_sub(sub, [1], a.taus, a.tol, 0.0)[1])
    for d in data.values():
        d["Xg"] = torch.tensor(d["X"], device=dev)
        d["yg"] = torch.tensor(d["y"], device=dev)
        d["swg"] = torch.tensor(d["sw"], device=dev)
        d["rdg"] = torch.tensor(d["rd"], device=dev)
        d["wg"] = torch.tensor(d["w"], device=dev)
        d["axg"] = torch.tensor(d["ax"], device=dev)
    print(f"datos listos ({time.time() - t0:.0f}s): {len(data)} secuencias de entrenamiento, dispositivo {dev}", flush=True)
    folds = [] if a.only_all else (a.folds or mids)
    for seed in a.model_seeds:
        out = {sub: {} for sub in a.test}
        for mid in list(folds) + (["ALL"] if (a.all or a.only_all) else []):
            seqs = [d for k, d in data.items() if k[2] != mid]
            mu_, sd_ = stats(seqs)
            mu, sd = torch.tensor(mu_, device=dev), torch.tensor(sd_, device=dev)
            model = train(seqs, mu, sd, a, seed, dev)
            if mid == "ALL":
                torch.save({"state": model.state_dict(), "mu": mu_, "sd": sd_, "args": vars(a)},
                           os.path.join(OUT, "models", f"{a.tag}_m{seed}.pt"))
                print(f"  [{a.tag} m{seed}] ALL guardado ({time.time() - t0:.0f}s)", flush=True)
                continue
            for sub in a.test:
                for h in (0, 1):
                    dn, dw = test[(sub, 1, mid, False, h)], test[(sub, 1, mid, True, h)]
                    pn, sn = predict(model, dn["X"], mu, sd, dev)
                    pw, sw_ = predict(model, dw["X"], mu, sd, dev)
                    p = 0.5 * (pn + pw[:, :, [1, 0, 2]])
                    s = 0.5 * (sn + sw_)
                    out[sub][f"{mid}_P{h}"] = np.repeat(p, STRIDE, axis=0)[:dn["n"]].astype(np.float16)
                    out[sub][f"{mid}_S{h}"] = np.repeat(s, STRIDE, axis=0)[:dn["n"]].astype(np.float16)
            print(f"  [{a.tag} m{seed}] {mid} listo ({time.time() - t0:.0f}s)", flush=True)
        for sub in a.test:
            if out[sub]:
                np.savez_compressed(os.path.join(OUT, f"{a.tag}_m{seed}__{sub}.npz"), taus=np.array(a.taus), **out[sub])
    print(f"listo ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
