"""
Utilidades compartidas por bs_auto.py, bs_sklearn.py y bs_numpy.py.

Esto NO es una de las "tres versiones" pedidas en el enunciado (esas se
diferencian por como resamplean/ajustan cada bootstrap); es solo I/O y
calculo del intervalo de confianza, para no repetir ese codigo 3 veces.
"""
import csv
import platform
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DATA_DIR / "results"

# Semilla para el RESAMPLEO bootstrap (paso 2.i del enunciado). Es DISTINTA
# de la semilla usada en generate_data.py (esa es para generar los datos;
# esta es para decidir que indices caen en cada resample). La dejamos fija
# y COMPARTIDA entre bs_sklearn.py y bs_numpy.py para que ambas resampleen
# exactamente los mismos B conjuntos de indices y sean comparables "manzana
# con manzana" en la parte (c) (la unica diferencia entre ellas pasa a ser
# el metodo de ajuste, no el azar). bs_auto.py resamplea con el generador
# interno de sklearn (via random_state), asi que sus indices no calzan uno
# a uno con estas dos, aunque el resultado estadistico deberia ser similar.
RESAMPLE_SEED = 123


def load_data():
    """Carga X, y, beta_true generados por generate_data.py."""
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    beta_true = np.load(DATA_DIR / "beta_true.npy")
    return X, y, beta_true


def resample_seeds(B, base_seed=RESAMPLE_SEED):
    """B semillas independientes (una por resample), derivadas de base_seed
    con numpy.random.SeedSequence (asi cada worker de joblib tiene su propio
    generador, sin overlap de streams entre resamples)."""
    ss = np.random.SeedSequence(base_seed)
    return ss.spawn(B)


def bootstrap_ci(beta_samples, alpha=0.05):
    """beta_samples: array (B, k+1). Devuelve (lower, upper), cada uno (k+1,).
    Percentil bootstrap: paso 3 del enunciado (se descarta 2.5% inferior y
    2.5% superior de cada columna)."""
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
