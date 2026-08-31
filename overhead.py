"""
Tarea 1 - IIC3533. Parte (h). Overhead T_o(p).

Usa los mismos tiempos T(p) de la parte (f), sin correr nada de nuevo, y
calcula el overhead que se definio en la clase 7.

    T0(p) = p * Tp menos T1

T0 igual a 0 es el caso ideal, todo el tiempo de los p procesos es trabajo
util. T0 mayor a 0 es tiempo que se gasto en coordinacion, esperas o
trabajo repetido, y no en avanzar el calculo.

Uso.
    python overhead.py --hostname LAPTOP-DJ126R18
"""
import argparse

from common import RESULTS_DIR
from speedup_efficiency import cargar_medianas


def main():
    parser = argparse.ArgumentParser(description="Calcula el overhead T0(p) a partir del csv de benchmark.py")
    parser.add_argument("--hostname", required=True,
                         help="nombre del computador, para leer results/benchmark_<hostname>.csv")
    args = parser.parse_args()

    csv_path = RESULTS_DIR / f"benchmark_{args.hostname}.csv"
    medianas, versiones, pmax = cargar_medianas(csv_path)

    for v in versiones:
        t1 = medianas[(v, 1)]
        print(f"\n{v}, T(1) es {t1:.3f} s")
        print(f"{'p':>3} {'T(p)':>10} {'p * T(p)':>10} {'T0(p)':>10} {'T0 / T1':>10}")
        for p in range(1, pmax + 1):
            tp = medianas[(v, p)]
            trabajo_total = p * tp
            t0 = trabajo_total - t1
            print(f"{p:>3} {tp:>10.3f} {trabajo_total:>10.3f} {t0:>10.3f} {t0 / t1:>10.3f}")


if __name__ == "__main__":
    main()
