"""
Tarea 1 - IIC3533. Parte (b): bootstrapping paralelo, version "auto".

El bootstrapping (generar los resamples) Y el paralelismo son enteramente
internos a sklearn.ensemble.BaggingRegressor(n_jobs=p): nosotros no
generamos los resamples ni llamamos a joblib directamente (aunque por
dentro BaggingRegressor SI usa joblib.Parallel, a traves del wrapper de
paralelismo de sklearn).

Iteracion: se probo copy_X=False en el estimador base (igual que en
bs_sklearn.py) pero se DESCARTO: a diferencia de bs_sklearn.py (que arma un
Xb = X[idx] nuevo por resample), BaggingRegressor reutiliza internamente el
mismo buffer de X entre estimadores (pasa un sample_weight con las cuentas
del bootstrap en vez de duplicar filas), asi que copy_X=False deja que cada
.fit() modifique ESE buffer compartido in-place -> se detecto empiricamente
(std de los coeficientes bootstrap ~20x mas alto de lo esperado, resultados
distintos entre p=1 y p=8) y se revirtio a copy_X=True (default). Buena
leccion para la parte (b): "itere y comente sus mejoras" tambien significa
descartar una optimizacion que resulta ser incorrecta, no solo quedarse con
la mas rapida. Se mantiene threadpool_limits(threads) alrededor del .fit(),
que si se verifico que da resultados identicos a variar solo la velocidad.

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
        estimator=LinearRegression(fit_intercept=False),  # copy_X=True (default): ver nota arriba
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
