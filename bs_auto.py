"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "auto".

El bootstrapping (generar los resamples) Y el paralelismo son enteramente
internos a sklearn.ensemble.BaggingRegressor(n_jobs=p): nosotros no
generamos los resamples ni llamamos a joblib directamente (aunque por
dentro BaggingRegressor SI usa joblib.Parallel).

Uso:
    python bs_auto.py --p 4 --B 48
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from common import load_data, summarize, save_timing

VERSION = "bs_auto"


def main():
    parser = argparse.ArgumentParser(description="Bootstrap con BaggingRegressor (paralelismo interno)")
    parser.add_argument("--p", type=int, default=1, help="n_jobs de BaggingRegressor")
    parser.add_argument("--B", type=int, default=48, help="n_estimators = numero de resamples")
    args = parser.parse_args()

    X, y, beta_true = load_data()

    beta_hat_full = LinearRegression(fit_intercept=False).fit(X, y).coef_

    # bootstrap=True + max_samples=1.0 => cada estimador se entrena con N
    # observaciones sorteadas CON reemplazo (igual al Paso 2 del enunciado).
    # El resampleo lo genera sklearn con su propio RNG interno (random_state
    # de aca abajo), NO con common.resample_seeds como en las otras 2
    # versiones, asi que sus indices no calzan uno a uno con bs_sklearn/bs_numpy
    # (comentar esto en la parte c).
    bagging = BaggingRegressor(
        estimator=LinearRegression(fit_intercept=False),
        n_estimators=args.B,
        max_samples=1.0,
        bootstrap=True,
        n_jobs=args.p,
        random_state=123,
    )

    t0 = time.perf_counter()
    bagging.fit(X, y)
    elapsed = time.perf_counter() - t0

    beta_samples = np.array([est.coef_ for est in bagging.estimators_])  # (B, k+1)

    summarize(VERSION, args.p, args.B, elapsed, beta_hat_full, beta_samples, beta_true)
    save_timing(VERSION, args.p, args.B, elapsed)


if __name__ == "__main__":
    main()
