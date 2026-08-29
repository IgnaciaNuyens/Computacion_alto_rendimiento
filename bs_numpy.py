"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "numpy".

Mismo esquema de paralelismo que bs_sklearn.py (joblib.Parallel(n_jobs=p),
MISMAS semillas de resampleo -> ver common.RESAMPLE_SEED), pero el ajuste de
cada resample resuelve (X^T X) beta = X^T y directamente con numpy, sin
pasar por sklearn.

Historial de iteracion (numeros medidos en este computador, p=8, B=48):
  v0: beta = np.linalg.inv(X^T X) @ (X^T y)         -> 2.013 s
  v1: beta = np.linalg.solve(X^T X, X^T y)           -> 0.956 s  (~2.1x)
      solve() hace LU + sustitucion, sin construir la inversa explicita
      (que es O(k^3) extra y menos estable numericamente). Formula del
      enunciado (X^T X)^-1 X^T y es matematicamente correcta, pero invertir
      a mano nunca es la forma eficiente de resolverla.
  v2 (version final, esta): se agrega threadpool_limits(threads) alrededor
      del bloque paralelo. Sin esto, cada uno de los p procesos abre ADEMAS
      sus propios threads de BLAS (en este equipo, MKL usa 4 threads por
      defecto) -> con p=8 se llega a tener ~32 threads compitiendo por 8
      cores (oversubscription, ver parte e). Con threadpool_limits(1) y
      p=8: 1.857 s -> 1.465 s (~21% mas rapido). Ver README para el detalle.

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
    """Resuelve (X^T X) beta = X^T y con np.linalg.solve (ver v0 vs v1
    arriba: es ~2x mas rapido que invertir X^T X explicitamente, ademas de
    mas estable numericamente)."""
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

    # Paso 1: beta_hat sobre el dataset completo (fuera del bloque medido).
    beta_hat_full = fit_normal_equations(X, y)

    # Paso 2: B resamples en paralelo. threadpool_limits(args.threads) evita
    # oversubscription: limita cuantos threads de BLAS puede abrir CADA
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
