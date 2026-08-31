"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "auto".

Acá el bootstrapping (generar los resamples) y el paralelismo los maneja
sklearn.ensemble.BaggingRegressor(n_jobs=p) por dentro, nosotros no
generamos los resamples ni llamamos a joblib directamente.

Al principio probamos copy_X=False en el estimador base, igual que en
bs_sklearn.py, pero acá dio resultados raros (la varianza de los
coeficientes bootstrap quedaba mucho mas alta de lo esperado). Quedo
documentado en el README, parte b, por que pasa eso, y por que en
bs_sklearn.py si es seguro usarlo pero acá no. Lo dejamos en el valor por
defecto (copy_X=True).

Uso:
    python bs_auto.py --p 4 --B 48 --threads 1
"""
import time

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from common import build_argparser, load_data, summarize, save_timing

VERSION = "bs_auto"


def main():
    parser = build_argparser("Bootstrap con BaggingRegressor (paralelismo interno)")
    parser.description += " (--B aqui es n_estimators)"
    args = parser.parse_args()

    X, y, beta_true = load_data()

    beta_hat_full = LinearRegression(fit_intercept=False).fit(X, y).coef_

    # bootstrap=True + max_samples=1.0 => cada estimador se entrena con N
    # observaciones sorteadas CON reemplazo (igual al Paso 2 del enunciado).
    # El resampleo lo genera sklearn con su propio RNG interno (random_state
    # de aca abajo), NO con common.resample_seeds como en bs_sklearn/bs_numpy,
    # asi que sus indices no calzan uno a uno con esas dos (comentado en la
    # parte c del informe).
    bagging = BaggingRegressor(
        estimator=LinearRegression(fit_intercept=False),  # copy_X=True por defecto, ver nota arriba
        n_estimators=args.B,
        max_samples=1.0,
        bootstrap=True,
        n_jobs=args.p,
        random_state=123,
    )

    with threadpool_limits(limits=args.threads):
        t0 = time.perf_counter()
        bagging.fit(X, y)
        elapsed = time.perf_counter() - t0

    beta_samples = np.array([est.coef_ for est in bagging.estimators_])  # (B, k+1)

    summarize(VERSION, args.p, args.B, elapsed, beta_hat_full, beta_samples, beta_true)
    save_timing(VERSION, args.p, args.B, elapsed)


if __name__ == "__main__":
    main()
