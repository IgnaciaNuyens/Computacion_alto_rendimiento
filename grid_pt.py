"""
Tarea 1 - IIC3533. Parte (i). Grilla de procesos p y threads t con p por t
menor o igual a p_max.

Corre una version del bootstrap para todas las combinaciones de p y t que
cumplen p por t menor o igual a p_max, y muestra cual da el mejor tiempo.
Sirve para responder de forma directa lo que las partes e y h dejaron
abierto, para un numero fijo de hilos de trabajo, conviene mas repartir
en threads internos o en procesos.

Uso.
    python grid_pt.py --pmax 8 --B 48 --repeticiones 3
    python grid_pt.py --pmax 8 --B 48 --repeticiones 3 --version bs_sklearn
"""
import argparse
import statistics
import time

from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from threadpoolctl import threadpool_limits

from common import load_data, resample_seeds
import bs_sklearn
import bs_numpy


def combinaciones_validas(pmax):
    """Todas las parejas (p, t) con p por t menor o igual a pmax."""
    combos = []
    for p in range(1, pmax + 1):
        t = 1
        while p * t <= pmax:
            combos.append((p, t))
            t += 1
    return combos


def medir(version, X, y, seeds, B, p, t, repeticiones):
    tiempos = []
    fit_fn = bs_numpy.fit_resample if version == "bs_numpy" else bs_sklearn.fit_resample
    for _ in range(repeticiones):
        if version == "bs_auto":
            bagging = BaggingRegressor(
                estimator=LinearRegression(fit_intercept=False),
                n_estimators=B, max_samples=1.0, bootstrap=True,
                n_jobs=p, random_state=123,
            )
            with threadpool_limits(limits=t):
                t0 = time.perf_counter()
                bagging.fit(X, y)
                tiempos.append(time.perf_counter() - t0)
        else:
            with threadpool_limits(limits=t):
                t0 = time.perf_counter()
                Parallel(n_jobs=p)(delayed(fit_fn)(X, y, s) for s in seeds)
                tiempos.append(time.perf_counter() - t0)
    return statistics.median(tiempos)


def main():
    parser = argparse.ArgumentParser(description="Grilla de p y t con p por t menor o igual a p_max")
    parser.add_argument("--pmax", type=int, default=8, help="p_max, tope para p por t")
    parser.add_argument("--B", type=int, default=48, help="numero de resamples bootstrap")
    parser.add_argument("--repeticiones", type=int, default=3,
                         help="repeticiones por combinacion, se usa la mediana")
    parser.add_argument("--version", default="bs_numpy", choices=["bs_numpy", "bs_sklearn", "bs_auto"])
    args = parser.parse_args()

    X, y, beta_true = load_data()
    seeds = resample_seeds(args.B)
    combos = combinaciones_validas(args.pmax)

    print(f"version {args.version}, {len(combos)} combinaciones de p y t con p por t hasta {args.pmax}")
    print(f"{'p':>3} {'t':>3} {'p por t':>8} {'tiempo (s)':>11}")
    resultados = []
    for p, t in combos:
        segundos = medir(args.version, X, y, seeds, args.B, p, t, args.repeticiones)
        resultados.append((p, t, segundos))
        print(f"{p:>3} {t:>3} {p * t:>8} {segundos:>11.3f}")

    mejor = min(resultados, key=lambda r: r[2])
    print()
    print(f"mejor combinacion, p igual a {mejor[0]}, t igual a {mejor[1]}, tiempo {mejor[2]:.3f} s")


if __name__ == "__main__":
    main()
