"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "numpy".

Mismo esquema de paralelismo que bs_sklearn.py (joblib.Parallel(n_jobs=p),
MISMAS semillas de resampleo -> ver common.RESAMPLE_SEED), pero el ajuste de
cada resample resuelve (X^T X) beta = X^T y directamente con numpy, sin
pasar por sklearn. Es la version pensada para optimizar "a mano".

Uso:
    python bs_numpy.py --p 4 --B 48
"""
import argparse
import time

import numpy as np
from joblib import Parallel, delayed

from common import load_data, resample_seeds, summarize, save_timing

VERSION = "bs_numpy"


def fit_normal_equations(X, y):
    """Resuelve (X^T X) beta = X^T y.
    np.linalg.solve es mas rapido y numericamente mas estable que calcular
    la inversa de X^T X explicitamente (que es lo que sugiere la formula
    beta_hat = (X^T X)^-1 X^T y del enunciado, pero nunca conviene invertir
    a mano)."""
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def fit_resample(X, y, seed):
    """Ajusta UN resample bootstrap con ecuaciones normales. Corre dentro
    de cada proceso worker de joblib."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=X.shape[0])
    Xb, yb = X[idx], y[idx]
    return fit_normal_equations(Xb, yb)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap con joblib.Parallel + ecuaciones normales (numpy)")
    parser.add_argument("--p", type=int, default=1, help="n_jobs para joblib.Parallel")
    parser.add_argument("--B", type=int, default=48, help="numero de resamples")
    args = parser.parse_args()

    X, y, beta_true = load_data()

    # Paso 1: beta_hat sobre el dataset completo.
    beta_hat_full = fit_normal_equations(X, y)

    # Paso 2: B resamples en paralelo.
    seeds = resample_seeds(args.B)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=args.p)(
        delayed(fit_resample)(X, y, s) for s in seeds
    )
    elapsed = time.perf_counter() - t0
    beta_samples = np.array(results)  # (B, k+1)

    summarize(VERSION, args.p, args.B, elapsed, beta_hat_full, beta_samples, beta_true)
    save_timing(VERSION, args.p, args.B, elapsed)


if __name__ == "__main__":
    main()
