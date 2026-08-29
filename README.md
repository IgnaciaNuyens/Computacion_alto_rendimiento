# Tarea 1 — IIC3533 Computación de Alto Rendimiento (2026-2)

Bootstrapping paralelo para regresión lineal con `joblib`. Grupo de 3 personas,
experimentos en al menos 2 computadores distintos. **Entrega: viernes 11 de
septiembre 2026, 23:59.**

## Estado actual

- [x] **(a) Generación de datos** — `generate_data.py` (hecho por Ignacia)
- [x] **(b) Tres implementaciones** — `bs_auto.py`, `bs_sklearn.py`, `bs_numpy.py` (+ `common.py` con utilidades compartidas), ya iteradas y con la bitácora de mejoras documentada abajo. Falta pasar esto al informe en PDF.
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
common.py                # (b) - utilidades compartidas: load_data, argparser, IC bootstrap, guardar tiempos
bs_auto.py                 # (b) - listo: BaggingRegressor(n_jobs=p)
bs_sklearn.py                # (b) - listo: joblib.Parallel + LinearRegression
bs_numpy.py                    # (b) - listo: joblib.Parallel + ecuaciones normales con numpy puro
benchmark.py                     # (f) - TODO: corre las 3 versiones para p=1..p_max y guarda tiempos
oversubscription.py                # (e) - TODO: threadpool_info() variando p
grid_pt.py                           # (i) - TODO: grid (p, t) con threadpool_limits
results/                               # csv/json de tiempos, por máquina (ver mas abajo)
  timings.csv                            # se genera solo, cada corrida de bs_*.py agrega una fila (gitignored)
plots/                                   # TODO: figuras para el informe
informe/                                   # TODO: fuente del informe (o link a Overleaf en este README)
```

## Parte (b): bitácora de iteración (para el "comente sus mejoras" del enunciado)

Todo esto se corrió en el mismo computador (8 cores lógicos), datos de
`generate_data.py`, B=48. Guarden esta sección para el informe.

**1. `bs_numpy.py`: invertir a mano vs. `np.linalg.solve`.**
La fórmula del enunciado es β̂ = (XᵀX)⁻¹Xᵀy, pero calcular la inversa
explícita es más caro (factorización + un producto matricial extra) y menos
estable que resolver el sistema directamente.

| Variante | Tiempo (p=8, t=1, B=48) |
|---|---|
| `np.linalg.inv(XtX) @ Xty` (v0, literal del enunciado) | 2.013 s |
| `np.linalg.solve(XtX, Xty)` (v1, versión final) | 0.956 s (**~2.1x más rápido**) |

**2. Oversubscription: `threadpoolctl.threadpool_limits`.**
Con `threadpool_info()` (parte e) se detectó que el NumPy de este entorno
usa **Intel MKL con 4 threads por proceso por defecto**. Eso significa que
al lanzar `joblib.Parallel(n_jobs=8)`, cada uno de los 8 procesos abre
además sus propios threads de BLAS: se puede llegar a ~32 threads
compitiendo por 8 cores físicos/lógicos (oversubscription). Al envolver el
bloque paralelo en `threadpool_limits(limits=t)` y fijar `t=1` para `p=8`:

| Config (`bs_numpy.py`, B=48) | Tiempo |
|---|---|
| p=8, sin limitar threads | 1.857 s |
| p=8, `threadpool_limits(1)` | 1.465 s (**~21% más rápido**) |

Esto se agregó a las 3 versiones vía el flag `--threads`/`-t` (ver
`common.build_argparser`), que además es justo lo que pide la parte (i)
más adelante (grid de `p` × `t`).

**3. `copy_X=False`: optimización que se descartó por incorrecta.**
Se probó `LinearRegression(copy_X=False)` para evitar que sklearn haga una
copia interna extra de cada resample. En `bs_sklearn.py` es seguro (cada
resample ya es un array `Xb = X[idx]` nuevo y descartable) — resultados
idénticos con y sin el flag, se mantiene. Pero en `bs_auto.py`
(`BaggingRegressor`) **rompía los resultados**: la varianza de los
coeficientes bootstrap se disparaba ~20x (0.01 → 0.12–0.25) y los
resultados cambiaban entre p=1 y p=8, porque `BaggingRegressor` no arma un
`Xb` nuevo por estimador — reutiliza el mismo buffer de `X` y pasa un
`sample_weight` con las cuentas del bootstrap, así que `copy_X=False` deja
que cada `.fit()` modifique ese buffer compartido in-place. Se revirtió a
`copy_X=True` (default) en `bs_auto.py`. Buena anécdota para el informe:
"iterar y comentar mejoras" también implica descartar una optimización que
resultó ser un bug, no solo quedarse con la más rápida.

**Tiempos finales (versión iterada, B=48, en este computador):**

| Versión | p=1, t=8 | p=8, t=1 |
|---|---|---|
| `bs_numpy` | 1.25 s | 2.15 s |
| `bs_sklearn` | 4.69 s | 7.01 s |
| `bs_auto` | 5.67 s | 11.55 s |

Ojo: en **esta** máquina (WSL2, 8 cores lógicos compartidos con el host de
Windows), p=1 con threads=8 salió más rápido que p=8 con threads=1 para las
3 versiones — el overhead de crear 8 procesos supera lo que se gana
paralelizando 48 tareas relativamente cortas. Esto es justo el tipo de
resultado que hay que reportar (no forzar) en las partes (f)-(h): el
overhead puede dominar dependiendo del tamaño del problema y del hardware.
Prueben esto también en el otro computador del grupo para la parte (j).

`bs_numpy` es consistentemente la más rápida de las 3 (ecuaciones normales
directas), luego `bs_sklearn` (resuelve por SVD vía `scipy.linalg.lstsq`,
más robusto pero más caro), y `bs_auto` es la más lenta (mismo ajuste que
`bs_sklearn` pero con el overhead extra de `BaggingRegressor`).

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
