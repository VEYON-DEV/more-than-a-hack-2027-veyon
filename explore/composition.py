"""Composición de color para que la pelota resalte: "separar los blancos".

Idea: los objetos claros del campo (pelota, camiseta lila del Athletic, líneas, medias, carteles)
parecen todos "blancos", pero en el plano de color a*b* (Lab) apuntan a direcciones distintas:

    pelota   ~ neutro            (a* -5, b* +7)
    Athletic ~ lila/magenta      (a* > 0, b* < 0)
    líneas   ~ verdoso           (a* < 0, b* > 0, se mezclan con el césped)

La composición:
  1. Solo píxeles claros (L >= L_MIN). El resto se pinta de un color plano (fondo).
  2. A los claros se les resta el color típico de la pelota y se amplifica su tinte K veces:
     lo que se parece a la pelota queda BLANCO, y cada "otro blanco" se vuelve un color fuerte
     distinto (Athletic -> violeta/azul, líneas -> verde/amarillo, piel -> naranja...).
  3. Fuera del campo (homografía, con margen) se apaga, para quitar carteles y tribuna.

Además calcula un mapa de "parecido a pelota" (claro y neutro) y cuenta cuántos candidatos
quedan por frame con color + tamaño esperado (según homografía) + forma redonda.

    .venv/bin/python explore/composition.py

Salida en data/derived/composition/ (gitignored: derivado del video).
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

P = asset_paths()
L_M, W_M = 105.3, 68.0
OUT = os.path.join(ROOT, "data", "derived", "composition")
os.makedirs(OUT, exist_ok=True)

# Color típico de la pelota en a*b*: mediana de los blobs que color_probe.py encontró
# automáticamente junto a la pelota del ATD (no viene de ningún frame elegido a mano).
A0, B0 = -5.0, 7.0
L_MIN = 165       # solo para estirar el brillo en la salida (OpenCV Lab, 0-255)
DL_MIN = 28       # "claro" = al menos esto más brillante que el césped de alrededor
CHROMA_MAX = 32   # y no demasiado saturado (el césped claro tiene croma ~40+)
K = 6.0           # cuánto se amplifica el tinte de los claros
SIGMA_AB = 6.0    # tolerancia de color para el mapa de parecido a pelota
PINK = (203, 160, 255)   # BGR, fondo pedido
BLACK = (25, 25, 25)


def load_h():
    df = pd.read_csv(P["homography"])
    return df[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)


def proj(h, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:]


def pitch_mask(h, shape, margin_m=2.0):
    mx, my = margin_m / L_M, margin_m / W_M
    corners = np.array([[-mx, -my], [1 + mx, -my], [1 + mx, 1 + my], [-mx, 1 + my]])
    # densify edges so the polygon survives perspective
    t = np.linspace(0, 1, 30)[:, None]
    poly = np.vstack([corners[i] + t * (corners[(i + 1) % 4] - corners[i]) for i in range(4)])
    q = proj(h, poly).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [q], 1)
    return m.astype(bool)


def local_background(L):
    """Brillo del césped alrededor: mediana grande sobre una versión reducida (rápido)."""
    small = cv2.resize(L.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, 11)  # ~44 px en resolución completa
    return cv2.resize(bg, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def bright_mask(L, a, b):
    """Claro respecto del entorno y poco saturado: independiente del sol/sombra y de las franjas."""
    dL = L - local_background(L)
    return (dL >= DL_MIN) & (np.hypot(a - A0, b - B0) < CHROMA_MAX), dL


def composition(img, bg=PINK, mask=None):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    bright, _ = bright_mask(L, a, b)
    a2 = np.clip((a - A0) * K, -127, 127) + 128
    b2 = np.clip((b - B0) * K, -127, 127) + 128
    L2 = np.interp(L, [120, 255], [170, 255])
    out = cv2.cvtColor(np.dstack([L2, a2, b2]).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out[~bright] = bg
    if mask is not None:
        out[~mask] = (out[~mask] * 0.25).astype(np.uint8)
    return out


def ball_score(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    _, dL = bright_mask(L, a, b)
    light = np.clip((dL - 20) / 60, 0, 1)
    neutral = np.exp(-((a - A0) ** 2 + (b - B0) ** 2) / (2 * SIGMA_AB ** 2))
    return light * neutral


def candidates(score, h, mask, thr=0.45):
    """Blobs claros+neutros, del tamaño esperado de la pelota en esa zona, y redondos."""
    m = (score > thr) & mask
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    out = []
    hinv = np.linalg.inv(h)
    for i in range(1, n):
        cx, cy = cent[i]
        g = proj(hinv, (cx, cy))[0]
        u1 = proj(h, g + [1 / L_M, 0])[0]
        ppm = np.hypot(*(u1 - (cx, cy)))
        d = 0.22 * ppm
        exp_area = np.pi * d * d / 4
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (0.2 * exp_area <= area <= 2.5 * exp_area + 4):
            continue
        if max(bw, bh) > 2.2 * max(1, min(bw, bh)):
            continue
        out.append((cx, cy, ppm))
    return out


def main():
    ball = np.load(os.path.join(ROOT, "data", "derived", "ball_atd.npy"))
    H = load_h()
    cap = cv2.VideoCapture(P["video"])

    def read(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        return img if ok else None

    # 1) imágenes: frame completo + zoom alrededor de la pelota
    show = [54416, 131796, 172179, 23416]
    for f in show:
        img = read(f)
        m = pitch_mask(H[f], img.shape)
        comp_pink = composition(img, PINK, m)
        comp_black = composition(img, BLACK, m)
        cv2.imwrite(os.path.join(OUT, f"frame_{f}_composicion.jpg"), comp_pink, [cv2.IMWRITE_JPEG_QUALITY, 88])
        bx, by = proj(H[f], ball[f])[0]
        r = 90
        cx, cy = int(bx), int(by - 40)
        x0, y0 = max(0, cx - r), max(0, cy - r)
        sl = (slice(y0, y0 + 2 * r), slice(x0, x0 + 2 * r))
        sc = (np.clip(ball_score(img)[sl], 0, 1) * 255).astype(np.uint8)
        panels = [img[sl], comp_pink[sl], comp_black[sl], cv2.cvtColor(sc, cv2.COLOR_GRAY2BGR)]
        panels = [cv2.resize(p, (360, 360), interpolation=cv2.INTER_NEAREST) for p in panels]
        for p, t in zip(panels, ["original", "composicion (fondo rosado)", "composicion (fondo negro)",
                                 "parecido a pelota"]):
            cv2.putText(p, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(p, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(OUT, f"frame_{f}_zoom.png"), np.hstack(panels))
        print("wrote frame", f)

    # 2) ¿cuántos candidatos deja el color? (40 frames con pelota en el ATD)
    rng = np.random.default_rng(1)
    graded = np.zeros(len(ball), bool)
    graded[11185:90652] = True
    graded[110808:190198] = True
    idx = np.flatnonzero(graded & ~np.isnan(ball[:, 0]))
    frames = np.sort(rng.choice(idx, 40, replace=False))
    n_c, hit, nearest_rank = [], 0, []
    for f in frames:
        img = read(f)
        if img is None:
            continue
        m = pitch_mask(H[f], img.shape)
        c = candidates(ball_score(img), H[f], m)
        n_c.append(len(c))
        bx, by = proj(H[f], ball[f])[0]
        # ATD queda ~0.5-1 m hacia la cámara y más si la pelota va por el aire: tolerancia 3 m
        ok = [np.hypot(cx - bx, cy - by) < 3.0 * ppm for cx, cy, ppm in c]
        hit += any(ok)
    n_c = np.array(n_c)
    print(f"\ncandidatos por frame (color + tamaño + forma, dentro del campo): "
          f"mediana {np.median(n_c):.0f}, p90 {np.percentile(n_c, 90):.0f}, max {n_c.max()}")
    print(f"frames donde algún candidato está a <3 m de la pelota del ATD: {hit}/{len(n_c)}")


if __name__ == "__main__":
    main()
