"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "sklearn".

El paralelismo lo armamos nosotros con joblib.Parallel(n_jobs=p), nosotros
generamos los B resamples y lanzamos un joblib.delayed por resample. El
ajuste de cada resample lo hace sklearn.linear_model.LinearRegression.

Uso:
    python bs_sklearn.py --p 4 --B 48 --threads 1
"""
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from common import build_argparser, load_data, resample_seeds, summarize, save_timing

VERSION = "bs_sklearn"


def fit_resample(X, y, seed):
    """Ajusta LinearRegression sobre UN resample bootstrap. Esta funcion es
    la que corre dentro de cada proceso worker de joblib."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=X.shape[0])
    Xb, yb = X[idx], y[idx]
    # fit_intercept=False porque X ya trae la columna de 1's (parte a), si
    # no sklearn sumaria otro intercepto ademas del que ya esta en X.
    # copy_X=False porque Xb ya es una copia nueva (X[idx] siempre copia),
    # asi que no hace falta que sklearn copie de nuevo por dentro.
    model = LinearRegression(fit_intercept=False, copy_X=False)
    model.fit(Xb, yb)
    return model.coef_


def main():
    parser = build_argparser("Bootstrap con joblib.Parallel + sklearn LinearRegression")
    args = parser.parse_args()

    X, y, beta_true = load_data()

    # Paso 1 del enunciado, beta_hat sobre el dataset completo (sin resamplear).
    beta_hat_full = LinearRegression(fit_intercept=False).fit(X, y).coef_

    # Paso 2, B resamples, cada uno con su propia semilla independiente.
    # threadpool_limits evita oversubscription (ver bs_numpy.py / README).
    seeds = resample_seeds(args.B)
    with threadpool_limits(limits=args.threads):
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
