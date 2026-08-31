"""
Tarea 1 - IIC3533. Parte (d). Como crea procesos el backend multiprocessing
de joblib.

Este script no explica con palabras, muestra con datos reales que pasa
adentro de joblib.Parallel cuando corremos bs_sklearn.py o bs_numpy.py
con distintos valores de p.

Cada tarea (cada resample bootstrap) devuelve el pid del proceso que la
ejecuto y el tipo de arreglo que le llego como X. Con eso alcanza para
ver dos cosas, cuantos procesos usa joblib de verdad, y que hace con
un arreglo grande como X.

Uso.
    python inspect_workers.py --p 4 --B 12
"""
import argparse
import os

import numpy as np
from joblib import Parallel, delayed

from common import load_data


def inspect(task_id, X, y):
    return {
        "task": task_id,
        "pid": os.getpid(),
        "tipo_X": type(X).__name__,
        "tipo_y": type(y).__name__,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspecciona los procesos worker de joblib")
    parser.add_argument("--p", type=int, default=4, help="n_jobs para joblib.Parallel")
    parser.add_argument("--B", type=int, default=12, help="cantidad de tareas a lanzar")
    args = parser.parse_args()

    X, y, beta_true = load_data()
    print("pid del proceso principal", os.getpid())
    print("tamano de X en MB", round(X.nbytes / 1e6, 2))
    print("tamano de y en MB", round(y.nbytes / 1e6, 2))
    print()

    resultados = Parallel(n_jobs=args.p)(delayed(inspect)(i, X, y) for i in range(args.B))
    for r in resultados:
        print(r)

    pids_usados = sorted(set(r["pid"] for r in resultados))
    print()
    print(f"pediste p={args.p}, joblib uso {len(pids_usados)} procesos worker distintos", pids_usados)
    print(f"lanzaste {args.B} tareas, cada proceso worker resolvio varias por turnos")


if __name__ == "__main__":
    main()
