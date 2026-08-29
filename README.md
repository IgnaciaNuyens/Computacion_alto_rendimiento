# Tarea 1 — IIC3533 Computación de Alto Rendimiento (2026-2)

Bootstrapping paralelo para regresión lineal con `joblib`. Grupo de 3 personas,
experimentos en al menos 2 computadores distintos. **Entrega: viernes 11 de
septiembre 2026, 23:59.**

## Estado actual

- [x] **(a) Generación de datos** — `generate_data.py` (hecho por Ignacia)
- [x] **(b) Tres implementaciones** — `bs_auto.py`, `bs_sklearn.py`, `bs_numpy.py` (+ `common.py` con utilidades compartidas). Faltaría redactar en el informe el "comente sus distintas mejoras" con números reales de varias corridas.
- [ ] **(c) Correctitud y reproducibilidad**
- [ ] **(d) Cómo crea procesos el backend `multiprocessing` de joblib**
- [ ] **(e) Oversubscription con `threadpoolctl`**
- [ ] **(f) Tiempos T(p) para p = 1..p_max, 3 versiones**
- [ ] **(g) Speedup S(p) y eficiencia E(p)**
- [ ] **(h) Overhead T_o(p)**
- [ ] **(i) Grid de (procesos p) x (threads t) con p·t ≤ p_max**
- [ ] **(j) Comparación entre los 2 computadores**
- [ ] Informe final en PDF

## Cómo generar los datos

```
python generate_data.py
```

Genera `beta_true.npy`, `X.npy`, `y.npy` (N=10 000, k=300, semilla fija = 42).
**No hace falta subir estos `.npy` al repo**: como la semilla está fija y
`numpy.random.default_rng` es determinista (no depende del hardware), correr
el script en cualquiera de los 2 computadores del grupo produce *exactamente*
los mismos datos. Está en `.gitignore` por eso. Si en algún momento dudan de
que dos máquinas dieron lo mismo, comparen con `np.array_equal` o un hash
(`np.save` + `sha256sum`).

## Entorno

```
conda create -n tarea1-hpc python=3.13 -y
conda activate tarea1-hpc
conda install numpy matplotlib joblib threadpoolctl scikit-learn -y
```

(El enunciado no lista `scikit-learn` en el comando de instalación, pero
`bs_auto.py` y `bs_sklearn.py` lo necesitan — `BaggingRegressor` y
`LinearRegression` son de ahí.)

Instalen el mismo entorno en ambos computadores del grupo (misma versión de
Python/numpy si es posible, para que los tiempos sean comparables).

## Estructura del repo (propuesta)

```
generate_data.py       # (a) - listo
bs_auto.py              # (b) - BaggingRegressor(n_jobs=p)
bs_sklearn.py            # (b) - joblib.Parallel + LinearRegression
bs_numpy.py               # (b) - joblib.Parallel + solución normal con numpy puro
benchmark.py             # corre las 3 versiones para p=1..p_max y guarda tiempos
oversubscription.py      # (e) - threadpool_info() variando p
grid_pt.py                # (i) - grid (p, t) con threadpool_limits
results/                  # csv/json de tiempos, por máquina (ver mas abajo)
  results_pc1.csv
  results_pc2.csv
plots/                    # figuras para el informe
informe/                   # fuente del informe (o link a Overleaf en este README)
```

## Cómo lo vamos a dividir (propuesta, ajusten si quieren)

La tarea tiene dependencias: (b) hay que tenerlo antes que casi todo lo
demás. Por eso conviene que **una persona termine (b) primero** y las otras
dos avancen en paralelo con las partes que **no** dependen de tener código
corriendo:

1. **Persona 1 (Ignacia)** — ya hizo (a). Sigue con **(b)**: las 3
   implementaciones. Sube cada script apenas funcione, aunque no esté
   optimizado (después se itera).
2. **Persona 2** — puede partir ya con **(d)** (es teórico, no necesita
   código: cómo `multiprocessing` crea procesos, fork vs spawn, copia de
   memoria, qué pasa con `X` e `y` al lanzar p procesos) y dejar listo
   **benchmark.py** (el script que corre cada versión para p=1..p_max y mide
   tiempos), que solo se puede probar de verdad cuando (b) esté listo.
3. **Persona 3** — arma el esqueleto de los gráficos con `matplotlib` para
   T(p)/S(p)/E(p)/overhead (partes f, g, h) usando datos de prueba
   (inventados) mientras (b) no está listo, así cuando lleguen los tiempos
   reales solo hay que enchufar los datos.

Cuando (b) esté en el repo:
- Cada persona corre `benchmark.py` **en su propio computador** (así se
  cumple el requisito de "al menos 2 computadores distintos") y sube su
  `results/results_<nombre>.csv`.
- Con esos 2+ archivos de resultados se arman los gráficos de (f), (g), (h)
  y la comparación de (j).
- (c) y (e) se escriben una vez que hay resultados reales para comentar.
- (i) lo hace quien terminó (b), usando la versión más eficiente de las 3.

## Flujo de trabajo con git

```
git clone <URL-del-repo>
git checkout -b nombre-de-quien-trabaja   # opcional, o directo a main si prefieren simple
# ... trabajan ...
git add archivo.py
git commit -m "mensaje corto"
git pull --rebase origin main
git push origin main
```

Como somos pocos y el código son scripts chicos, no hace falta un flujo muy
formal: **avisar por el grupo antes de tocar un archivo que otro está
editando**, comentar el `push` en el chat, y listo.

## Dónde compartimos el código

GitHub (repo privado). Pasos para quien lo cree:

1. Crear un repo privado en github.com (ej. `tarea1-hpc-iic3533`).
2. Agregar a los otros 2 integrantes como colaboradores (Settings → Collaborators),
   con su usuario de GitHub.
3. Conectar esta carpeta local:
   ```
   git remote add origin https://github.com/<usuario>/tarea1-hpc-iic3533.git
   git add .
   git commit -m "parte (a): generacion de datos"
   git push -u origin main
   ```
