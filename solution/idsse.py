"""Lector de IDSSE (DFL / Bundesliga 2022-23, CC-BY 4.0) -> nuestro formato.

Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event data in
elite soccer." Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
Archivos: https://huggingface.co/datasets/pysport/idsse-data (copiados en data/external/idsse/)

Son OTROS partidos (no Barça-Athletic): usarlos para validar y entrenar está permitido por
RULES.md ("Data of this kind about other games is fine").

Por cada partido y tiempo guarda (data/derived/solution/idsse/<match>.npz):
  players  (n, K, 2)  metros, origen en una esquina (x en [0, 105], y en [0, 68]); NaN = ausente
  teams    (K,)       0 = local, 1 = visitante
  gk       (K,)       True si es arquero (PlayingPosition == 'TW')
  ball     (n, 2)     metros
  label    (n,)       0 = H (local con la pelota), 1 = A, 2 = D (BallStatus == 0)

    .venv/bin/python solution/idsse.py
"""

import glob
import os
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, WORK  # noqa: E402

SRC = os.path.join(ROOT, "data", "external", "idsse")
OUT = os.path.join(WORK, "idsse")
os.makedirs(OUT, exist_ok=True)


def read_meta(path):
    root = ET.parse(path).getroot()
    g = root.find(".//General")
    env = root.find(".//Environment")
    home, guest = g.get("HomeTeamId"), g.get("GuestTeamId")
    gk, team_of = set(), {}
    for team in root.iter("Team"):
        for p in team.iter("Player"):
            team_of[p.get("PersonId")] = team.get("TeamId")
            if p.get("PlayingPosition") == "TW":
                gk.add(p.get("PersonId"))
    return {"home": home, "guest": guest, "L": float(env.get("PitchX")), "W": float(env.get("PitchY")),
            "gk": gk, "title": g.get("MatchTitle"), "team_of": team_of}


def convert(match_id):
    meta_p = glob.glob(os.path.join(SRC, f"DFL_02_01_matchinformation_*{match_id}.xml"))[0]
    pos_p = glob.glob(os.path.join(SRC, f"DFL_04_03_positions_raw_observed_*{match_id}.xml"))[0]
    meta = read_meta(meta_p)
    L, W = meta["L"], meta["W"]
    sets = {}                                   # (section, team, person) -> lists
    cur = None
    for ev, el in ET.iterparse(pos_p, events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            key = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
            cur = sets.setdefault(key, {"n": [], "x": [], "y": [], "pos": [], "st": []})
        elif ev == "end" and el.tag == "Frame" and cur is not None:
            a = el.attrib
            cur["n"].append(int(a["N"]))
            cur["x"].append(float(a["X"]))
            cur["y"].append(float(a["Y"]))
            if "BallStatus" in a:
                cur["pos"].append(int(a.get("BallPossession", 0)))
                cur["st"].append(int(a["BallStatus"]))
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            cur = None
            el.clear()
    out = {"title": meta["title"]}
    persons = sorted({k[2] for k in sets if k[1] in (meta["home"], meta["guest"])},
                     key=lambda p: (meta["team_of"].get(p) != meta["home"], p))
    slot = {p: i for i, p in enumerate(persons)}
    teams = np.array([0 if meta["team_of"].get(p) == meta["home"] else 1 for p in persons], np.int8)
    gk = np.array([p in meta["gk"] for p in persons])
    for h, section in enumerate(("firstHalf", "secondHalf")):
        bkey = [k for k in sets if k[0] == section and k[1].upper() == "BALL"][0]
        b = sets[bkey]
        n0, n1 = min(b["n"]), max(b["n"])
        n = n1 - n0 + 1
        ball = np.full((n, 2), np.nan, np.float32)
        lab = np.full(n, 2, np.int8)
        bi = np.array(b["n"]) - n0
        ball[bi, 0] = np.array(b["x"]) + L / 2
        ball[bi, 1] = W / 2 - np.array(b["y"])
        st, ps = np.array(b["st"]), np.array(b["pos"])
        lab[bi] = np.where(st == 0, 2, np.where(ps == 1, 0, 1))
        pl = np.full((n, len(persons), 2), np.nan, np.float32)
        for (sec, team, person), d in sets.items():
            if sec != section or person not in slot:
                continue
            idx = np.array(d["n"]) - n0
            ok = (idx >= 0) & (idx < n)
            pl[idx[ok], slot[person], 0] = np.array(d["x"])[ok] + L / 2
            pl[idx[ok], slot[person], 1] = W / 2 - np.array(d["y"])[ok]
        out[f"players_{h}"] = pl
        out[f"ball_{h}"] = ball
        out[f"label_{h}"] = lab
    out["teams"], out["gk"] = teams, gk
    out["pitch"] = np.array([L, W], np.float32)
    np.savez_compressed(os.path.join(OUT, f"{match_id}.npz"), **out)
    l0, l1 = out["label_0"], out["label_1"]
    lab = np.r_[l0, l1]
    return (f"{match_id} {meta['title']}: {len(lab)} frames  H {100 * (lab == 0).mean():.0f}%  "
            f"A {100 * (lab == 1).mean():.0f}%  D {100 * (lab == 2).mean():.0f}%  jugadores {len(persons)}")


def main():
    ids = sorted({os.path.basename(p).split("DFL-MAT-")[1][:-4]
                  for p in glob.glob(os.path.join(SRC, "DFL_04_03_*.xml"))})
    with Pool(len(ids)) as pool:
        for line in pool.imap_unordered(convert, ids):
            print(line, flush=True)


if __name__ == "__main__":
    main()
