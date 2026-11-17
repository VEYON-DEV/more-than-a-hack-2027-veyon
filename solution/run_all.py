"""Regenera la entrega completa desde los datos del partido (data/tactical/, data/broadcast/).

    python3 -m venv .venv       && .venv/bin/pip install -r requirements-solution.txt
    python3 -m venv .venv-torch && .venv-torch/bin/pip install -r requirements-torch.txt
    .venv/bin/python solution/run_all.py --out evaluation/results/<equipo>.csv

Pasos:
  1. get_idsse.py / idsse.py   descarga y convierte IDSSE (otros partidos, CC-BY 4.0)
  2. ball_candidates.py        candidatos a pelota en el video táctico
  3. broadcast_sync.py         sincroniza el tracking del broadcast con el táctico (sin video)
     ball_track.py             trayectoria de la pelota: Kalman + suavizado RTS (táctico + video + broadcast)
  4. explore/entity_colors.py  colores de camiseta por equipo (desde ATD + video, sin etiquetas)
  5. team_check.py             verificación del equipo de cada track por color
  6. build_bench.py            IDSSE ensuciado como nuestro tracking (v3: huecos de pelota según el estado
                               del juego, jugadores suavizados) -> pistas y etiquetas de entrenamiento
  7. apply.py (1.ª pasada)     exporta las pistas por frame de Barça-Athletic (bar_feats.npz)
  8. paradas_apply.py predict  12 redes temporales multi-definición (entorno PyTorch), vista normal y
                               espejada, entrenadas con IDSSE v3 + paradas simuladas
  9. paradas_apply.py csv      cabeza de 1.5 s, Viterbi modulado por la probabilidad de cambio de la red,
                               duración mínima y reglas seguras -> CSV
 10. validate_submission.py    mismas comprobaciones que el evaluador oficial

Pesos guardados en solution/models (las GPU no son bit a bit deterministas):
  paradas/cc3all_m{1..12}.pt   las 12 redes
  ball_gaps_by_state.npz       largos de huecos de la pelota con juego vivo / muerto en nuestro tracking
  ball_src_runs.npz            largos de rachas de la pelota medida por el video
Con --retrain las redes se vuelven a entrenar (mismo resultado salvo el no determinismo de la GPU).
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 1.ª pasada de apply.py: exporta las pistas por frame (sus parámetros de decodificación no afectan la exportación)
FEATS = ["--mirror", "--seeds", "1", "2", "3", "--tau-train", "1.5", "--tau", "0.4", "--alpha", "2.0", "--beta", "0.6",
         "--mindur", "1.5", "--g-sw", "1.5", "--g-rs", "0"]
TAG = "cc3all"                    # 12 redes multi-definición
SEEDS = list(range(1, 13))
HEAD = "1.5"                      # definición de posesión de la entrega: control mínimo de 1.5 s
DEGRADE = "v3"


def run(py, args, env=None):
    print("\n$", py, " ".join(args), flush=True)
    subprocess.run([py] + args, cwd=ROOT, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("evaluation", "results", "solucion.csv"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--torch-python", default=os.path.join(ROOT, ".venv-torch", "bin", "python"))
    ap.add_argument("--skip-perception", action="store_true",
                    help="no rehacer los pasos 1-5 si sus salidas ya existen (solo para pruebas rápidas)")
    ap.add_argument("--retrain", action="store_true",
                    help="reentrenar las 12 redes en vez de usar los pesos guardados (lento, usa GPU)")
    a = ap.parse_args()
    py = sys.executable
    W = os.path.join("data", "derived", "solution")
    env = dict(os.environ, DEGRADE=DEGRADE)
    if not a.skip_perception:
        run(py, ["solution/get_idsse.py"])
        run(py, ["solution/idsse.py"])
        run(py, ["solution/ball_candidates.py", "--workers", str(a.workers)])
        run(py, ["solution/broadcast_sync.py"])          # pelota del broadcast, sincronizada sin video
        run(py, ["solution/ball_track.py"])
        run(py, ["explore/entity_colors.py"])
        run(py, ["solution/team_check.py", "--workers", str(a.workers)])
    run(py, ["solution/build_bench.py"], env)          # pistas de IDSSE ensuciado, en paralelo
    run(py, ["solution/apply.py"] + FEATS + ["--export-feats", f"{W}/bar_feats.npz", "--out", f"{W}/tmp_sin_red.csv"], env)

    tags = [f"{TAG}_m{s}" for s in SEEDS]
    models_dir = os.path.join(HERE, "models", "paradas")
    if a.retrain:
        run(py, ["solution/paradas_ballz.py"])           # altura de la pelota de IDSSE
        run(py, ["solution/paradas_ctrl.py"])            # etiquetas por contacto (cabezas auxiliares)
        run(py, ["solution/paradas_build.py", "--variant", "carry", "--pc", "0.5", "--ret", "0.3", "--suffix", "v3r"], env)
        run(a.torch_python, ["solution/paradas_tcn.py", "--tag", TAG, "--train", "bench_fcarry_p0.5v3r", "bench_v3",
                             "--test", "bench_v3", "--taus", "0", "0.5", "1.0", "1.5", "3.0", "-1", "-2", "-3",
                             "--head-w", "0.5", "1", "1", "1", "1", "1", "1", "1", "--aux", "1.0", "--only-all",
                             "--model-seeds"] + [str(s) for s in SEEDS], env)
        models_dir = os.path.join(W, "paradas", "models")
    probs = f"{W}/bar_probs.npz"
    run(a.torch_python, ["solution/paradas_apply.py", "predict", "--tags"] + tags +
        ["--feats", f"{W}/bar_feats.npz", "--models-dir", models_dir, "--out", probs], env)
    run(py, ["solution/paradas_apply.py", "csv", "--probs", probs, "--head", HEAD, "--out", a.out], env)
    run(py, ["evaluation/validate_submission.py", a.out])


if __name__ == "__main__":
    main()
