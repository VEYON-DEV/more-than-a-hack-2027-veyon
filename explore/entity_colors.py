"""Estima automáticamente el color de cada entidad para la composición v3.

  barca / athletic      k-means sobre torsos de jugadores de campo del ATD (equipo por track)
  gk_barca / gk_athletic  torsos de los 2 tracks que el ATD marca como 'Goalkeeper'
  referee               blobs de tamaño persona dentro del campo, a >2.5 m de todo jugador
                        del ATD (el ATD no trae árbitros): su cluster dominante

Nada se etiqueta a mano: los colores salen del ATD + video. Guarda
data/derived/entity_colors.json y mosaicos de recortes para comprobar a ojo.

    .venv/bin/python explore/entity_colors.py
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "explore"))
from games import asset_paths  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from composition import load_h, proj, pitch_mask, L_M, W_M  # noqa: E402

P = asset_paths()
OUT = os.path.join(ROOT, "data", "derived")
CROPS = os.path.join(OUT, "composition")


def goalkeeper_slots():
    root = ET.parse(P["atd_meta"]).getroot()
    out = []
    for slot, pl in enumerate(root.iter("Player")):
        for prm in pl.iter("ProviderParameter"):
            if prm.findtext("Name") == "position_type" and prm.findtext("Value") == "Goalkeeper":
                out.append(slot)
    return out


def not_grass_mask(img, thr=22):
    """True donde el píxel se aleja del césped LOCAL (no solo por tono: los arqueros van de
    verde y los árbitros de amarillo flúor, que un filtro por tono confunde con césped)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 11), (img.shape[1], img.shape[0]))
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.linalg.norm(lab - lab_bg, axis=-1) > thr


def kmeans(data, k):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, lbl, centers = cv2.kmeans(data.astype(np.float32), k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lbl.ravel(), minlength=k) / len(lbl)
    order = np.argsort(-share)
    return centers[order], share[order]


def player_box(h, x, y):
    u, v = proj(h, (x, y))[0]
    ppm = np.hypot(*(proj(h, (x + 1 / L_M, y))[0] - (u, v)))
    hpx = 1.8 * ppm * 0.75
    x0, x1 = int(u - 0.22 * hpx), int(u + 0.22 * hpx)
    y0 = int(v - 0.4 * ppm - hpx)
    return x0, y0, x1, int(v - 0.4 * ppm), ppm


def load_ball_atd():
    """Pelota del ATD táctico (coordenadas normalizadas, NaN si falta), por frame de video.
    Misma lectura que explore/probe_frames.py; se guarda en data/derived/ball_atd.npy."""
    path = os.path.join(OUT, "ball_atd.npy")
    if os.path.exists(path):
        return np.load(path)
    ball = np.full((199319, 2), np.nan, dtype=np.float32)
    with open(P["atd_raw"]) as fh:
        for line in fh:
            fid, _, bl = line.rstrip("\n").split(":")
            if bl[0] != "N":
                bx, by, _ = bl.split(",")
                ball[int(fid) - 1] = (float(bx), float(by))   # video_frame = frame_id - 1
    os.makedirs(OUT, exist_ok=True)
    np.save(path, ball)
    return ball


def main():
    ball = load_ball_atd()
    H = load_h()
    teams = team_of_each_track(P["atd_meta"])
    gks = goalkeeper_slots()
    print("goalkeeper slots:", gks, "teams:", [("barca", "athletic")[teams[s]] for s in gks])

    rng = np.random.default_rng(3)
    graded = np.zeros(len(ball), bool)
    graded[11185:90652] = True
    graded[110808:190198] = True
    frames = sorted(int(f) for f in rng.choice(np.flatnonzero(graded), 60, replace=False))
    want = set(frames)
    atd = {}
    with open(P["atd_raw"]) as fh:
        for line in fh:
            fid = int(line[:line.index(":")]) - 1
            if fid in want:
                atd[fid] = line.split(":")[1]

    cap = cv2.VideoCapture(P["video"])
    pix = {k: [] for k in ("barca", "athletic", "gk_barca", "gk_athletic", "referee")}
    crops = {k: [] for k in ("gk_barca", "gk_athletic", "referee")}
    grass_ref = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        h = H[f]
        ng = not_grass_mask(img)
        gl = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[pitch_mask(h, img.shape, 0.0) & ~ng]
        grass_ref.append(np.median(gl.astype(float), axis=0)[None])
        feet = []
        for slot, g in enumerate(atd[f].split(";")):
            if g[0] == "N":
                continue
            x, y, _ = map(float, g.split(","))
            feet.append((x, y))
            x0, y0, x1, y1, ppm = player_box(h, x, y)
            if x0 < 0 or y0 < 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
                continue
            ty1 = int(y0 + 0.45 * (y1 - y0))
            torso = img[y0:ty1, x0:x1][ng[y0:ty1, x0:x1]]
            team = ("barca", "athletic")[teams[slot]]
            key = ("gk_" + team) if slot in gks else team
            pix[key].append(torso)
            if slot in gks and len(crops[key]) < 20:
                c = img[max(0, y0 - 5):y1 + 5, max(0, x0 - 10):x1 + 10]
                if c.size:
                    crops[key].append(cv2.resize(c, (60, 100), interpolation=cv2.INTER_NEAREST))

        # referee: foreground blobs of person size, inside the pitch, far from every ATD player
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.medianBlur(small, 11), (img.shape[1], img.shape[0]))
        lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
        fg = (np.linalg.norm(lab - lab_bg, axis=-1) > 22) & pitch_mask(h, img.shape, margin_m=0.0)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, comp, stats, cent = cv2.connectedComponentsWithStats(fg)
        hinv = np.linalg.inv(h)
        feet = np.array(feet) if feet else np.zeros((0, 2))
        for i in range(1, n):
            x, y, w, hh, area = stats[i]
            gx, gy = proj(hinv, (x + w / 2, y + hh))[0]  # bottom of blob ~ feet
            ppm = np.hypot(*(proj(h, (gx + 1 / L_M, gy))[0] - proj(h, (gx, gy))[0]))
            if not (1.0 * ppm < hh < 2.6 * ppm and 0.25 * ppm < w < 1.4 * ppm):
                continue
            if len(feet):
                dm = np.min(np.hypot((feet[:, 0] - gx) * L_M, (feet[:, 1] - gy) * W_M))
                if dm < 2.5:
                    continue
            torso = img[y:y + int(0.5 * hh), x:x + w][comp[y:y + int(0.5 * hh), x:x + w] == i]
            sub_ng = ng[y:y + int(0.5 * hh), x:x + w][comp[y:y + int(0.5 * hh), x:x + w] == i]
            pix["referee"].append(torso[sub_ng])
            if len(crops["referee"]) < 20:
                crops["referee"].append(cv2.resize(img[y:y + hh, x:x + w], (60, 100), interpolation=cv2.INTER_NEAREST))

    clusters = {}
    for k, v in pix.items():
        data = np.vstack(v).astype(np.float32) / 255 if v else np.zeros((0, 3))
        if len(data) < 50:
            print(f"{k}: too few pixels ({len(data)})")
            continue
        clusters[k] = kmeans(data, 6)
    # a cluster is a SEED colour only if (a) no other entity has a cluster close to it and
    # (b) it is not close to grass (lit or shaded). Grass references: median of local grass in
    # the sampled frames (lit) and that colour darkened (shade).
    to_lab = lambda c: cv2.cvtColor(np.uint8([[np.clip(c, 0, 1) * 255]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    grass_lab = np.median(np.vstack(grass_ref), axis=0)
    grass_refs = [grass_lab, grass_lab * [0.75, 1, 1] + [0, 0, 0]]
    colors = {}
    for k, (centers, share) in clusters.items():
        keep = []
        for c, s_ in zip(centers, share):
            others = [o for kk, (oc, osh) in clusters.items() if kk != k for o, os_ in zip(oc, osh) if os_ > 0.06]
            dmin = min(np.linalg.norm(to_lab(c) - to_lab(o)) for o in others)
            dgrass = min(np.linalg.norm(to_lab(c)[1:] - g[1:]) + 0.5 * abs(to_lab(c)[0] - g[0]) for g in grass_refs)
            if s_ <= 0.06:
                tag = "minor"
            elif dgrass < 22:
                tag = "grass-like"
            elif dmin <= 24:
                tag = "shared"
            else:
                tag = "SEED"
            if tag == "SEED":
                keep.append(c.tolist())
            print(f"  {k:12} RGB{tuple(int(x * 255) for x in c[::-1])}  {s_:4.0%}  "
                  f"dmin_otro={dmin:5.1f}  dcesped={dgrass:5.1f}  {tag}")
        colors[k] = keep
    json.dump(colors, open(os.path.join(OUT, "entity_colors.json"), "w"), indent=1)
    for k, v in crops.items():
        if v:
            cv2.imwrite(os.path.join(CROPS, f"crops_{k}.png"), np.hstack(v[:20]))
    print("saved", os.path.join(OUT, "entity_colors.json"))


if __name__ == "__main__":
    main()
