"""
Tarea 1 - IIC3533. Parte (e). Oversubscription con threadpoolctl.

Mide que pasa cuando el numero de procesos p y el numero de threads
internos de BLAS t suman mas hilos que cores tiene la maquina.

Primero muestra que libreria de algebra lineal usa numpy y cuantos
threads abre por defecto (threadpool_info). Despues corre bs_numpy.py con
distintas combinaciones de p y t, y compara p por t contra la cantidad de
cores (os.cpu_count).

Uso.
    python oversubscription.py --B 48
"""
import argparse
import os
import statistics
import time

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

from common import load_data, resample_seeds
import bs_numpy


def tiempo_de(p, t, X, y, seeds, repeticiones=3):
    """Corre la misma configuracion varias veces y devuelve la mediana.
    Una sola corrida en esta maquina compartida (WSL2) tiene bastante
    ruido, con la mediana de 3 repeticiones el numero es mas confiable."""
    tiempos = []
    for _ in range(repeticiones):
        with threadpool_limits(limits=t):
            t0 = time.perf_counter()
            Parallel(n_jobs=p)(delayed(bs_numpy.fit_resample)(X, y, s) for s in seeds)
            tiempos.append(time.perf_counter() - t0)
    return statistics.median(tiempos)


def main():
    parser = argparse.ArgumentParser(description="Mide oversubscription variando p y t")
    parser.add_argument("--B", type=int, default=48, help="numero de resamples bootstrap")
    args = parser.parse_args()

    cores = os.cpu_count()
    print(f"cores logicos disponibles en esta maquina, os.cpu_count() {cores}")
    print()
    print("librerias de algebra lineal detectadas por threadpool_info")
    for lib in threadpool_info():
        print(f"  {lib.get('user_api')} / {lib.get('internal_api')}, threads por defecto {lib.get('num_threads')}")
    print()

    X, y, beta_true = load_data()
    seeds = resample_seeds(args.B)

    print("bs_numpy.py con distintas combinaciones de p (procesos) y t (threads por proceso)")
    print(f"{'p':>3} {'t':>3} {'p por t':>8} {'cores':>6} {'tiempo (s)':>11}")
    combinaciones = [
        (1, 1), (1, cores),
        (cores, 1), (cores, 2), (cores, 4), (cores, cores),
    ]
    for p, t in combinaciones:
        segundos = tiempo_de(p, t, X, y, seeds)
        marca = " sin oversubscription" if p * t <= cores else " con oversubscription"
        print(f"{p:>3} {t:>3} {p * t:>8} {cores:>6} {segundos:>11.3f}{marca}")


if __name__ == "__main__":
    main()
