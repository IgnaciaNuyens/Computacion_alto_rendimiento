"""
Funciones que usan bs_auto.py, bs_sklearn.py y bs_numpy.py, para no repetir
el mismo codigo en los 3 archivos (cargar los datos, leer los argumentos de
la linea de comandos, calcular el intervalo de confianza y guardar los
tiempos).
"""
import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DATA_DIR / "results"

# Semilla para decidir que filas caen en cada resample bootstrap (paso 2.i
# del enunciado). Es distinta de la semilla de generate_data.py, esa es
# para generar los datos, esta es para el resampleo. La dejamos fija y
# compartida entre bs_sklearn.py y bs_numpy.py para que las dos usen los
# mismos resamples y se puedan comparar en la parte (c).
RESAMPLE_SEED = 123


def build_argparser(description):
    """Argumentos que usan las 3 versiones, p (procesos), B (resamples) y
    t (threads internos de BLAS por proceso, para las partes e e i)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--p", type=int, default=1, help="numero de procesos (n_jobs de joblib)")
    parser.add_argument("--B", type=int, default=48, help="numero de resamples bootstrap")
    parser.add_argument("--threads", "-t", type=int, default=1,
                         help="threads internos de BLAS/OpenMP permitidos por proceso")
    return parser


def load_data():
    """Carga X, y, beta_true generados por generate_data.py."""
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    beta_true = np.load(DATA_DIR / "beta_true.npy")
    return X, y, beta_true


def resample_seeds(B, base_seed=RESAMPLE_SEED):
    """Una semilla distinta para cada uno de los B resamples, para que cada
    tarea use su propio generador de numeros aleatorios. Simplemente le
    sumamos el numero de resample a la semilla base."""
    return [base_seed + i for i in range(B)]


def bootstrap_ci(beta_samples, alpha=0.05):
    """Intervalo de confianza percentil, paso 3 del enunciado. beta_samples
    tiene forma (B, k+1), y para cada columna se descarta el 2.5% inferior
    y el 2.5% superior."""
    lower = np.percentile(beta_samples, 100 * alpha / 2, axis=0)
    upper = np.percentile(beta_samples, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def summarize(version, p, B, elapsed, beta_hat_full, beta_samples, beta_true):
    lower, upper = bootstrap_ci(beta_samples)
    covered = (beta_true >= lower) & (beta_true <= upper)
    print(f"\n=== {version} | p={p} | B={B} ===")
    print(f"Tiempo bootstrap (solo paso 2, sin contar carga de datos): {elapsed:.3f} s")
    print(f"Coeficientes cubiertos por su IC 95% (de {len(beta_true)}): {covered.sum()}")
    print("Primeros 5 coeficientes (beta_hat_full | IC95 | beta_true):")
    for j in range(min(5, len(beta_true))):
        print(f"  b{j}: {beta_hat_full[j]:7.3f}  "
              f"[{lower[j]:7.3f}, {upper[j]:7.3f}]  "
              f"true={beta_true[j]:7.3f}")
    return lower, upper, covered


def save_timing(version, p, B, elapsed):
    """Agrega una fila a results/timings.csv (insumo para las partes f, g, h, j)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "timings.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["version", "p", "B", "elapsed_seconds", "hostname", "timestamp"])
        writer.writerow([version, p, B, f"{elapsed:.6f}", platform.node(), time.time()])
