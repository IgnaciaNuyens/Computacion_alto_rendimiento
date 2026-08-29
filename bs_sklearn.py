"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "sklearn".

El paralelismo lo manejamos NOSOTROS con joblib.Parallel(n_jobs=p): nosotros
generamos los B resamples y lanzamos un joblib.delayed por resample. El
ajuste de cada resample lo hace sklearn.linear_model.LinearRegression.

Uso:
    python bs_sklearn.py --p 4 --B 48
"""
import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from common import load_data, resample_seeds, summarize, save_timing

VERSION = "bs_sklearn"


def fit_resample(X, y, seed):
    """Ajusta LinearRegression sobre UN resample bootstrap. Esta funcion es
    la que corre dentro de cada proceso worker de joblib."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=X.shape[0])
    Xb, yb = X[idx], y[idx]
    # X ya trae la columna de 1's agregada en generate_data.py, por eso
    # fit_intercept=False: si no, sklearn sumaria OTRO intercepto ademas
    # del que ya esta codificado como primera columna de X.
    model = LinearRegression(fit_intercept=False)
    model.fit(Xb, yb)
    return model.coef_


def main():
    parser = argparse.ArgumentParser(description="Bootstrap con joblib.Parallel + sklearn LinearRegression")
    parser.add_argument("--p", type=int, default=1, help="n_jobs para joblib.Parallel")
    parser.add_argument("--B", type=int, default=48, help="numero de resamples")
    args = parser.parse_args()

    X, y, beta_true = load_data()

    # Paso 1 del enunciado: beta_hat sobre el dataset completo (sin resamplear).
    beta_hat_full = LinearRegression(fit_intercept=False).fit(X, y).coef_

    # Paso 2: B resamples, cada uno con su propia semilla independiente.
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
