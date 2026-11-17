"""Genera en paralelo las pistas de IDSSE ensuciado (3 semillas, normal y espejado) del banco.
La versión del ensuciado la da la variable DEGRADE (v1 / v2, ver common.py).

    DEGRADE=v2 .venv/bin/python solution/build_bench.py
"""
import glob
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, BENCH_SUB  # noqa: E402


def job(args):
    from bench import match_features
    p, seed, swap = args
    match_features(p, seed=seed, swap=swap)
    return os.path.basename(p)[:-4], seed, swap


def main():
    os.makedirs(os.path.join(WORK, BENCH_SUB), exist_ok=True)
    jobs = [(p, s, w) for p in sorted(glob.glob(os.path.join(WORK, "idsse", "*.npz"))) for s in (1, 2, 3) for w in (False, True)]
    t = time.time()
    with Pool(int(os.environ.get("JOBS", "9"))) as pool:
        for m, s, w in pool.imap_unordered(job, jobs):
            print(f"  {m} s{s}{'w' if w else ''} ({time.time() - t:.0f}s)", flush=True)
    print(f"listo: {len(jobs)} archivos en {BENCH_SUB} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
