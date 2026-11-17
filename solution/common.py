"""Utilidades compartidas de la solución: rutas, homografía, geometría del campo, ATD.

Solo numpy + OpenCV + la librería estándar. Todo se deriva de los datos entregados.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

P = asset_paths()
N_FRAMES = 199319
FPS = 25
DT = 1.0 / FPS
L_M, W_M = 105.3, 68.0                      # games.py: medido del tracking corregido
WORK = os.path.join(ROOT, "data", "derived", "solution")   # gitignored (data/*)
# versión del ensuciado de IDSSE. v1: huecos de pelota al azar. v2: huecos según el estado del juego
# (vivo/muerto), medidos en Barça-Athletic (ball_gaps_by_state.py). Cada versión tiene su carpeta de banco.
DEGRADE = os.environ.get("DEGRADE", "v1")
BENCH_SUB = "bench" if DEGRADE == "v1" else f"bench_{DEGRADE}"
os.makedirs(WORK, exist_ok=True)


def halves():
    """[(half, first_video_frame, last_video_frame)] de los tramos calificados."""
    return half_video_blocks(GAME)


def load_homographies():
    import pandas as pd
    df = pd.read_csv(P["homography"])
    H = df[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy(np.float64).reshape(-1, 3, 3)
    return H


def proj(h, pts):
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:]


def ppm_at(h, hinv, u, v):
    """Píxeles por metro en el punto (u, v) de la imagen."""
    g = proj(hinv, (u, v))[0]
    return float(np.hypot(*(proj(h, g + [1 / L_M, 0])[0] - (u, v))))


def team_of_each_track():
    """0 = local (Barça), 1 = visitante (Athletic), por slot del ATD."""
    root = ET.parse(P["atd_meta"]).getroot()
    home = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)


def goalkeeper_slots():
    root = ET.parse(P["atd_meta"]).getroot()
    out = []
    for slot, pl in enumerate(root.iter("Player")):
        for prm in pl.iter("ProviderParameter"):
            if prm.findtext("Name") == "position_type" and prm.findtext("Value") == "Goalkeeper":
                out.append(slot)
    return out


def load_atd(cache=True):
    """players (N,46,2) y ball (N,2) en coordenadas normalizadas 0-1, NaN = ausente."""
    pp, bp = os.path.join(WORK, "atd_players.npy"), os.path.join(WORK, "atd_ball.npy")
    if cache and os.path.exists(pp) and os.path.exists(bp):
        return np.load(pp), np.load(bp)
    players = np.full((N_FRAMES, 46, 2), np.nan, np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, np.float32)
    with open(P["atd_raw"]) as fh:
        for line in fh:
            fid, pl, bl = line.rstrip("\n").split(":")
            vf = int(fid) - 1                       # DATA.md: video_frame = frame_id - 1
            for slot, g in enumerate(pl.split(";")):
                if g[0] != "N":
                    x, y, _ = g.split(",")
                    players[vf, slot] = (float(x), float(y))
            if bl[0] != "N":
                x, y, _ = bl.split(",")
                ball[vf] = (float(x), float(y))
    np.save(pp, players)
    np.save(bp, ball)
    return players, ball


def pitch_polygon(h, margin_m=0.0):
    import cv2  # noqa: F401
    mx, my = margin_m / L_M, margin_m / W_M
    c = np.array([[-mx, -my], [1 + mx, -my], [1 + mx, 1 + my], [-mx, 1 + my]])
    return proj(h, c).astype(np.int32)


def line_segments_m():
    """Líneas reglamentarias como lista de polilíneas en metros."""
    segs = []

    def seg(a, b, n=40):
        t = np.linspace(0, 1, n)[:, None]
        segs.append(np.array(a) + t * (np.array(b) - np.array(a)))

    seg((0, 0), (L_M, 0)); seg((L_M, 0), (L_M, W_M)); seg((L_M, W_M), (0, W_M)); seg((0, W_M), (0, 0))
    seg((L_M / 2, 0), (L_M / 2, W_M))
    for x0, s in ((0, 1), (L_M, -1)):
        for dx, hw in ((16.5, 20.16), (5.5, 9.16)):
            y0, y1 = W_M / 2 - hw, W_M / 2 + hw
            seg((x0, y0), (x0 + s * dx, y0)); seg((x0 + s * dx, y0), (x0 + s * dx, y1)); seg((x0 + s * dx, y1), (x0, y1))
    a = np.linspace(0, 2 * np.pi, 120)
    segs.append(np.c_[L_M / 2 + 9.15 * np.cos(a), W_M / 2 + 9.15 * np.sin(a)])
    half = np.arccos(5.5 / 9.15)
    for cx, s in ((11.0, 1), (L_M - 11.0, -1)):
        t = np.linspace(-half, half, 30)
        segs.append(np.c_[cx + s * 9.15 * np.cos(t), W_M / 2 + 9.15 * np.sin(t)])
    return segs


SPOTS_M = [(L_M / 2, W_M / 2), (11.0, W_M / 2), (L_M - 11.0, W_M / 2)]


def load_smart_tagging():
    """Lista de dict(code, start, end, team, type) con start/end en frames de video."""
    root = ET.parse(P["smart_tagging"]).getroot()
    out = []
    for inst in root.iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        s, e = float(inst.findtext("start")), float(inst.findtext("end"))
        tags = {}
        for lab in inst.iter("label"):
            g, t = (lab.findtext("group") or "").strip(), (lab.findtext("text") or "").strip()
            if g:
                tags[g.lower()] = t
        out.append({"code": code, "start": int(round(s * FPS)), "end": int(round(e * FPS)),
                    "team": tags.get("team", ""), "type": tags.get("type", ""), "tags": tags})
    return out
