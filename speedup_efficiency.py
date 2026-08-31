"""
Tarea 1 - IIC3533. Parte (g). Speedup S(p) y eficiencia E(p).

Lee el csv que dejo benchmark.py (parte f) y calcula, para cada version,
lo mismo que se definio en la clase 7.

    Sp = T1 / Tp
    Ep = Sp / p

No corre nada de nuevo, solo reusa los tiempos ya medidos. Usa la mediana
de las repeticiones guardadas para cada p, la misma idea de la parte e,
no confiar en una sola corrida.

Uso.
    python speedup_efficiency.py --hostname LAPTOP-DJ126R18
"""
import argparse
import csv
import statistics
from collections import defaultdict

from common import RESULTS_DIR


def cargar_medianas(csv_path):
    filas = list(csv.DictReader(open(csv_path)))
    grupos = defaultdict(list)
    for fila in filas:
        clave = (fila["version"], int(fila["p"]))
        grupos[clave].append(float(fila["segundos"]))
    medianas = {clave: statistics.median(tiempos) for clave, tiempos in grupos.items()}
    versiones = sorted(set(v for v, p in medianas))
    pmax = max(p for v, p in medianas)
    return medianas, versiones, pmax


def main():
    parser = argparse.ArgumentParser(description="Calcula speedup y eficiencia a partir del csv de benchmark.py")
    parser.add_argument("--hostname", required=True,
                         help="nombre del computador, para leer results/benchmark_<hostname>.csv")
    args = parser.parse_args()

    csv_path = RESULTS_DIR / f"benchmark_{args.hostname}.csv"
    medianas, versiones, pmax = cargar_medianas(csv_path)

    for v in versiones:
        t1 = medianas[(v, 1)]
        print(f"\n{v}, T(1) es {t1:.3f} s")
        print(f"{'p':>3} {'T(p)':>10} {'S(p)':>8} {'E(p)':>8}")
        for p in range(1, pmax + 1):
            tp = medianas[(v, p)]
            sp = t1 / tp
            ep = sp / p
            print(f"{p:>3} {tp:>10.3f} {sp:>8.3f} {ep:>8.3f}")


if __name__ == "__main__":
    main()
