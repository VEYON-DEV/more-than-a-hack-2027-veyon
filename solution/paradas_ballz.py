"""Altura de la pelota (Z) de IDSSE, que idsse.py no guarda. Lee solo las líneas de la pelota del XML.

    .venv/bin/python solution/paradas_ballz.py   ->  data/derived/solution/paradas/idsse_ballz.npz  (<partido>_z<h>)
"""
import glob
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, WORK  # noqa: E402

SRC = os.path.join(ROOT, "data", "external", "idsse")
RX = re.compile(r'N="(\d+)".*? Z="([-\d.]+)"')


def job(mid):
    path = glob.glob(os.path.join(SRC, f"DFL_04_03_positions_raw_observed_*{mid}.xml"))[0]
    txt = subprocess.run(["grep", "BallStatus=", path], capture_output=True, text=True).stdout
    nz = np.array([(int(a), float(b)) for a, b in RX.findall(txt)])
    cut = np.flatnonzero(np.diff(nz[:, 0]) > 1000)
    parts = np.split(nz, cut + 1)
    z = np.load(os.path.join(WORK, "idsse", f"{mid}.npz"))
    out = {}
    for h, part in enumerate(parts[:2]):
        n0 = int(part[:, 0].min())
        zz = np.full(len(z[f"ball_{h}"]), np.nan, np.float32)
        zz[part[:, 0].astype(int) - n0] = part[:, 1]
        out[f"{mid}_z{h}"] = zz
    return out


def main():
    mids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(WORK, "idsse", "*.npz")))
    save = {}
    with Pool(7) as pool:
        for o in pool.imap_unordered(job, mids):
            save.update(o)
    os.makedirs(os.path.join(WORK, "paradas"), exist_ok=True)
    np.savez_compressed(os.path.join(WORK, "paradas", "idsse_ballz.npz"), **save)
    for k in sorted(save):
        v = save[k]
        print(k, len(v), f"NaN {np.isnan(v).mean():.3f}  z>1.5m {np.nanmean(v > 1.5):.3f}  z>3m {np.nanmean(v > 3):.3f}  max {np.nanmax(v):.1f}")


if __name__ == "__main__":
    main()
