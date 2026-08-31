"""
Tarea 1 - IIC3533. Parte (c). Correctitud y reproducibilidad.

Corremos las tres versiones sobre los mismos datos y revisamos dos cosas.
Primero, si corremos la misma version dos veces con la misma semilla, nos
tiene que dar exactamente lo mismo (reproducibilidad). Segundo, si las
tres versiones terminan estimando mas o menos lo mismo aunque calculen
las cosas de forma distinta por dentro (correctitud).

Uso:
    python verify_correctness.py --p 4 --B 48
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from threadpoolctl import threadpool_limits

from common import build_argparser, load_data, resample_seeds, bootstrap_ci

import bs_sklearn
import bs_numpy


def run_manual(fit_fn, X, y, seeds, p, threads):
    """Corre bs_sklearn.fit_resample o bs_numpy.fit_resample en paralelo,
    sobre las mismas semillas, y devuelve los 48 vectores de coeficientes."""
    with threadpool_limits(limits=threads):
        results = Parallel(n_jobs=p)(delayed(fit_fn)(X, y, s) for s in seeds)
    return np.array(results)


def run_auto(X, y, B, p, threads, random_state=123):
    """Corre bs_auto.py (BaggingRegressor) y devuelve los 48 vectores de
    coeficientes, uno por estimador interno."""
    bagging = BaggingRegressor(
        estimator=LinearRegression(fit_intercept=False),
        n_estimators=B, max_samples=1.0, bootstrap=True,
        n_jobs=p, random_state=random_state,
    )
    with threadpool_limits(limits=threads):
        bagging.fit(X, y)
    return np.array([e.coef_ for e in bagging.estimators_])


def coverage(beta_samples, beta_true):
    lower, upper = bootstrap_ci(beta_samples)
    covered = (beta_true >= lower) & (beta_true <= upper)
    return covered.sum(), (upper - lower).mean()


def main():
    parser = build_argparser("Verifica correctitud y reproducibilidad de las 3 versiones")
    args = parser.parse_args()

    X, y, beta_true = load_data()
    seeds = resample_seeds(args.B)

    print("Paso 1. Reproducibilidad. Cada version se corre dos veces con los")
    print("mismos datos y las mismas semillas, y se comparan los resultados.\n")

    numpy_1 = run_manual(bs_numpy.fit_resample, X, y, seeds, args.p, args.threads)
    numpy_2 = run_manual(bs_numpy.fit_resample, X, y, seeds, args.p, args.threads)
    print("bs_numpy, dos corridas identicas ", np.array_equal(numpy_1, numpy_2))

    sklearn_1 = run_manual(bs_sklearn.fit_resample, X, y, seeds, args.p, args.threads)
    sklearn_2 = run_manual(bs_sklearn.fit_resample, X, y, seeds, args.p, args.threads)
    print("bs_sklearn, dos corridas identicas ", np.array_equal(sklearn_1, sklearn_2))

    auto_1 = run_auto(X, y, args.B, args.p, args.threads)
    auto_2 = run_auto(X, y, args.B, args.p, args.threads)
    print("bs_auto, dos corridas identicas ", np.array_equal(auto_1, auto_2))

    print("\nPaso 2. Correctitud entre bs_sklearn y bs_numpy. Usan los mismos")
    print("resamples, la unica diferencia es el metodo de ajuste, asi que sus")
    print("coeficientes deberian coincidir salvo error numerico de punto flotante.\n")
    diff = np.abs(sklearn_1 - numpy_1)
    print("diferencia absoluta maxima entre las dos versiones ", diff.max())
    print("diferencia absoluta promedio ", diff.mean())

    print("\nPaso 3. Correctitud estadistica de las tres versiones. bs_auto no")
    print("comparte resamples con las otras dos, asi que se compara a nivel")
    print("estadistico, cuantos de los 301 coeficientes reales caen dentro del")
    print("intervalo de confianza 95% de cada version, y que tan ancho es ese")
    print("intervalo en promedio.\n")
    for nombre, muestras in [("bs_numpy", numpy_1), ("bs_sklearn", sklearn_1), ("bs_auto", auto_1)]:
        cubiertos, ancho_promedio = coverage(muestras, beta_true)
        print(f"{nombre:10s} cubiertos {cubiertos} de 301, ancho promedio del IC {ancho_promedio:.5f}")


if __name__ == "__main__":
    main()
