"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "numpy".

Mismo esquema de paralelismo que bs_sklearn.py (joblib.Parallel con
n_jobs=p y las mismas semillas de resampleo, ver common.RESAMPLE_SEED),
pero acá el ajuste de cada resample no pasa por sklearn, se resuelve
directamente (X^T X) beta = X^T y con numpy.

Al principio lo habiamos hecho invirtiendo la matriz a mano, con
np.linalg.inv(X^T X) @ (X^T y), que es literalmente la formula del
enunciado. Nos dimos cuenta que np.linalg.solve(X^T X, X^T y) da lo mismo
y es bastante mas rapido (probamos ambas y quedo el detalle con los
tiempos en el README, parte b), asi que nos quedamos con solve.

Uso:
    python bs_numpy.py --p 4 --B 48 --threads 1
"""
import time

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from common import build_argparser, load_data, resample_seeds, summarize, save_timing

VERSION = "bs_numpy"


def fit_normal_equations(X, y):
    """Resuelve (X^T X) beta = X^T y con np.linalg.solve, en vez de calcular
    la inversa de X^T X a mano (ver docstring del archivo)."""
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
    parser = build_argparser("Bootstrap con joblib.Parallel + ecuaciones normales (numpy)")
    args = parser.parse_args()

    X, y, beta_true = load_data()

    # Paso 1, beta_hat sobre el dataset completo (fuera del bloque medido).
    beta_hat_full = fit_normal_equations(X, y)

    # Paso 2, B resamples en paralelo. threadpool_limits(args.threads) evita
    # oversubscription, limita cuantos threads de BLAS puede abrir cada
    # proceso worker (joblib propaga este limite automaticamente a los
    # procesos que lanza, no hace falta ponerlo dentro de fit_resample).
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
