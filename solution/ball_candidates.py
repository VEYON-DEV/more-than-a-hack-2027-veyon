"""Paso 1: candidatos a pelota en el video táctico, para todos los frames calificados.

Versión rápida de la detección de la composición (explore/composition_v4.py): sin segmentar
jugadores, solo manchas blancas del tamaño esperado de una pelota (según la homografía) dentro
del campo, fuera de las líneas y puntos pintados. Para cada candidato guarda rasgos que después
usa el filtro de Kalman para elegir:

  x_m, y_m   posición en el campo (metros), por homografía
  area_ratio área / área esperada de una pelota en ese punto
  fill       área / círculo mínimo que la contiene (1 = disco; media o botín << 1)
  contrast   cuánto más clara que el césped local
  ring_fg    fracción del anillo alrededor (más allá del halo) que no es césped (0 = aislada)

    .venv/bin/python solution/ball_candidates.py --workers 14

Salida: data/derived/solution/ball_candidates.npz
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P, WORK, L_M, W_M, halves, load_homographies, proj, ppm_at, pitch_polygon,  # noqa: E402
                    line_segments_m, SPOTS_M)

# color de la pelota en Lab de OpenCV (L 0-255, a/b +128): mediana medida automáticamente con
# explore/color_probe.py sobre blobs junto a la pelota del ATD -> (238, -5, +7)
BALL_L_MIN = 185
# caja asimétrica en a/b: la pelota borrosa se corre hacia el verde (a baja, b sube); el lila
# del Athletic (a ~133, b ~122) queda fuera
A_RANGE, B_RANGE = (108, 130), (127, 150)
CONTRAST_MIN = 28
FG_DE = 16
FIELDS = ("frame", "u", "v", "x_m", "y_m", "area", "area_ratio", "fill", "contrast", "ring_fg")

_H = None


def _init():
    global _H
    _H = load_homographies()


def frame_candidates(img, f, h):
    H_, W_ = img.shape[:2]
    hinv = np.linalg.inv(h)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    white = cv2.inRange(lab, (BALL_L_MIN, A_RANGE[0], B_RANGE[0]), (255, A_RANGE[1], B_RANGE[1]))

    mask = np.zeros((H_, W_), np.uint8)
    cv2.fillPoly(mask, [pitch_polygon(h, 2.0)], 1)
    excl = np.zeros((H_, W_), np.uint8)
    for s in line_segments_m():
        q = proj(h, s / [L_M, W_M]).astype(np.int32)
        cv2.polylines(excl, [q], False, 1, 9)
    for sx, sy in SPOTS_M:
        su, sv = proj(h, (sx / L_M, sy / W_M))[0]
        cv2.circle(excl, (int(su), int(sv)), max(3, int(0.35 * ppm_at(h, hinv, su, sv))), 1, -1)
    allowed = ((mask == 1) & (excl == 0)).astype(np.uint8) * 255
    white = cv2.bitwise_and(white, allowed)

    n, comp, stats, cent = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:
        return []
    small = cv2.resize(lab, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, 11)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 3 or area > 600 or max(w, hh) > 2.5 * max(1, min(w, hh)):
            continue
        cu, cv_ = cent[i]
        ppm = ppm_at(h, hinv, cu, cv_)
        exp_area = np.pi * (0.22 * ppm) ** 2 / 4
        ratio = area / max(exp_area, 1.0)
        if not (0.15 <= ratio <= 3.0):
            continue
        bgc = bg[min(bg.shape[0] - 1, int(cv_ / 4)), min(bg.shape[1] - 1, int(cu / 4))].astype(np.float32)
        blob_px = lab[y:y + hh, x:x + w][comp[y:y + hh, x:x + w] == i].astype(np.float32)
        contrast = float(blob_px[:, 0].max() - bgc[0])
        if contrast < CONTRAST_MIN:
            continue
        # redondez
        blob = (comp[y:y + hh, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), rad = cv2.minEnclosingCircle(cnts[0])
        fill = float(area / max(1.0, np.pi * (rad + 0.5) ** 2))
        # aislamiento: anillo entre r1 y r2 (más allá del halo borroso)
        r1 = max(3, int(0.20 * ppm))
        r2 = r1 + max(3, int(0.25 * ppm))
        y0, y1, x0, x1 = max(0, y - r2), min(H_, y + hh + r2), max(0, x - r2), min(W_, x + w + r2)
        b2 = (comp[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(b2, np.ones((2 * r2 + 1,) * 2, np.uint8)) &
                ~cv2.dilate(b2, np.ones((2 * r1 + 1,) * 2, np.uint8))).astype(bool)
        roi = lab[y0:y1, x0:x1].astype(np.float32)
        ring_fg = float((np.linalg.norm(roi[ring] - bgc, axis=1) > FG_DE).mean()) if ring.any() else 0.0
        gx, gy = proj(hinv, (cu, cv_))[0]
        out.append((f, cu, cv_, gx * L_M, gy * W_M, area, ratio, fill, contrast, ring_fg))
    return out


def process(chunk):
    f0, f1 = chunk
    cap = cv2.VideoCapture(P["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    rows = []
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        rows.extend(frame_candidates(img, f, _H[f]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--chunk", type=int, default=2500)
    ap.add_argument("--only", type=str, default=None, help="rango f0:f1 para probar")
    a = ap.parse_args()
    if a.only:
        f0, f1 = map(int, a.only.split(":"))
        spans = [(f0, f1)]
    else:
        spans = [(s, e) for _, s, e in halves()]
    chunks = []
    for s, e in spans:
        for c0 in range(s, e + 1, a.chunk):
            chunks.append((c0, min(e, c0 + a.chunk - 1)))
    t = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(process, chunks)):
            rows.extend(r)
            if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                print(f"  {i + 1}/{len(chunks)} bloques, {time.time() - t:.0f}s", flush=True)
    arr = np.array(rows, np.float32).reshape(-1, len(FIELDS))
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    out = os.path.join(WORK, "ball_candidates.npz" if not a.only else "ball_candidates_test.npz")
    np.savez_compressed(out, data=arr, fields=np.array(FIELDS))
    nf = sum(e - s + 1 for s, e in spans)
    print(f"{len(arr)} candidatos en {nf} frames ({len(arr) / nf:.1f}/frame) en {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
