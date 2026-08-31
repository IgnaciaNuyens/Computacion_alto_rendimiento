"""
Tarea 1 - IIC3533. Parte (f). Tiempos T(p) para las 3 versiones.

Corre bs_sklearn.py, bs_numpy.py y bs_auto.py (llamando directamente a sus
funciones, no por subprocess, para que sea mas rapido) para p desde 1
hasta p_max, con threads=1 fijo (el valor que dejamos como base en la
parte e), repite cada combinacion varias veces, y guarda TODAS las
repeticiones en un csv, no solo un promedio, para que en las partes g, h
y j se pueda recalcular todo sin volver a correr nada.

El csv de salida queda en results/benchmark_<nombre del computador>.csv,
asi que cuando cada integrante del grupo lo corra en su propia maquina,
los archivos no se pisan entre si.

Uso.
    python benchmark.py --pmax 8 --B 48 --repeticiones 3
"""
import argparse
import csv
import os
import platform
import time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

from common import load_data, resample_seeds, RESULTS_DIR
import bs_sklearn
import bs_numpy


def medir_manual(fit_fn, X, y, seeds, p, threads):
    """Sirve tanto para bs_sklearn.fit_resample como para bs_numpy.fit_resample."""
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        Parallel(n_jobs=p)(delayed(fit_fn)(X, y, s) for s in seeds)
        return time.perf_counter() - t0


def medir_bs_auto(X, y, B, p, threads, random_state=123):
    bagging = BaggingRegressor(
        estimator=LinearRegression(fit_intercept=False),
        n_estimators=B, max_samples=1.0, bootstrap=True,
        n_jobs=p, random_state=random_state,
    )
    with threadpool_limits(limits=threads):
        t0 = time.perf_counter()
        bagging.fit(X, y)
        return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Mide T(p) para las 3 versiones y lo guarda en un csv")
    parser.add_argument("--pmax", type=int, default=os.cpu_count(),
                         help="p maximo a probar, por defecto los cores de esta maquina")
    parser.add_argument("--B", type=int, default=48, help="numero de resamples bootstrap")
    parser.add_argument("--threads", type=int, default=1, help="threads internos de BLAS por proceso")
    parser.add_argument("--repeticiones", type=int, default=3,
                         help="cuantas veces repetir cada combinacion de version y p")
    args = parser.parse_args()

    X, y, beta_true = load_data()
    seeds = resample_seeds(args.B)
    hostname = platform.node()

    RESULTS_DIR.mkdir(exist_ok=True)
    salida = RESULTS_DIR / f"benchmark_{hostname}.csv"

    funciones = {
        "bs_sklearn": lambda p: medir_manual(bs_sklearn.fit_resample, X, y, seeds, p, args.threads),
        "bs_numpy": lambda p: medir_manual(bs_numpy.fit_resample, X, y, seeds, p, args.threads),
        "bs_auto": lambda p: medir_bs_auto(X, y, args.B, p, args.threads),
    }

    filas = []
    for version, funcion in funciones.items():
        for p in range(1, args.pmax + 1):
            for r in range(args.repeticiones):
                segundos = funcion(p)
                filas.append([version, p, args.threads, args.B, r, f"{segundos:.6f}", hostname])
                print(f"{version:10s} p={p} repeticion {r + 1} de {args.repeticiones}, {segundos:.3f} s")

    nuevo = not salida.exists()
    with open(salida, "a", newline="") as f:
        writer = csv.writer(f)
        if nuevo:
            writer.writerow(["version", "p", "threads", "B", "repeticion", "segundos", "hostname"])
        writer.writerows(filas)

    print()
    print("resultados guardados en", salida)


if __name__ == "__main__":
    main()
