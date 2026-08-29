"""
Tarea 1 - IIC3533 Computacion de Alto Rendimiento
Parte (a): Generacion de datos sinteticos para bootstrapping en regresion lineal.

Genera:
  - beta_true.npy : coeficientes verdaderos beta*  (shape: (k+1,))
  - X.npy         : matriz de diseno X, con columna de 1's al inicio (shape: (N, k+1))
  - y.npy         : vector de salidas y = X @ beta* + ruido  (shape: (N,))

IMPORTANTE: la semilla esta fija, asi que cada integrante del grupo puede
correr este script en su propio computador y obtener EXACTAMENTE los mismos
beta_true.npy / X.npy / y.npy (no hace falta compartir los .npy por el
repo, ver README.md). Las tres versiones (bs_auto.py, bs_sklearn.py,
bs_numpy.py) deben usar estos MISMOS archivos, no generar cada una sus
propios datos, para poder comparar resultados en la parte (c).
"""

import numpy as np

# ---- Parametros del experimento (fijos por el enunciado) ----
N = 10_000   # numero de observaciones
k = 300      # numero de variables de entrada
SEED = 42    # semilla fija para reproducibilidad

rng = np.random.default_rng(SEED)

# (i) Coeficientes verdaderos beta* ~ N(0,1), independientes.
#     Son k+1 porque incluyen el intercepto beta_0.
beta_true = rng.normal(loc=0.0, scale=1.0, size=k + 1)

# (ii) Matriz de datos: N filas, k columnas, entradas i.i.d. N(0,1).
#      Se le agrega una columna de 1's al inicio -> X queda de tamano N x (k+1).
X_raw = rng.normal(loc=0.0, scale=1.0, size=(N, k))
ones_column = np.ones((N, 1))
X = np.hstack([ones_column, X_raw])   # X ∈ R^(N x (k+1))

# (iii) y = X beta* + ruido, ruido ~ N(0,1) i.i.d. por observacion.
noise = rng.normal(loc=0.0, scale=1.0, size=N)
y = X @ beta_true + noise

# ---- Guardar para que todo el grupo use exactamente los mismos datos ----
np.save("beta_true.npy", beta_true)
np.save("X.npy", X)
np.save("y.npy", y)

print("Datos generados y guardados: beta_true.npy, X.npy, y.npy")
print(f"X shape: {X.shape}  (esperado: ({N}, {k+1}))")
print(f"y shape: {y.shape}  (esperado: ({N},))")
print(f"beta_true shape: {beta_true.shape}  (esperado: ({k+1},))")
