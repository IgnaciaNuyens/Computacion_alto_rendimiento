# Tarea 1 — IIC3533 Computación de Alto Rendimiento (2026-2)

Bootstrapping paralelo para regresión lineal con `joblib`. Grupo de 3 personas,
experimentos en al menos 2 computadores distintos. **Entrega: viernes 11 de
septiembre 2026, 23:59.**

## Estado actual

- [x] **(a) Generación de datos** — `generate_data.py` (hecho por Ignacia)
- [x] **(b) Tres implementaciones** — `bs_auto.py`, `bs_sklearn.py`, `bs_numpy.py` (+ `common.py` con utilidades compartidas), ya iteradas y con la bitácora de mejoras documentada abajo. Falta pasar esto al informe en PDF.
- [x] **(c) Correctitud y reproducibilidad** hecha con `verify_correctness.py`, resultados y explicación abajo.
- [x] **(d) Cómo crea procesos el backend `multiprocessing` de joblib** hecha con `inspect_workers.py`, resultados y explicación abajo.
- [x] **(e) Oversubscription con `threadpoolctl`** hecha con `oversubscription.py`, resultados y explicación abajo.
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
verify_correctness.py            # (c) - listo: reproducibilidad y correctitud entre las 3 versiones
inspect_workers.py                 # (d) - listo: muestra que hace joblib con los procesos worker
oversubscription.py                  # (e) - listo: threadpool_info() y tiempos variando p y t
benchmark.py                           # (f) - TODO: corre las 3 versiones para p=1..p_max y guarda tiempos
grid_pt.py                               # (i) - TODO: grid (p, t) con threadpool_limits
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

**2. Threads internos de BLAS, primera medición y por qué no bastaba con una sola corrida.**
Con `threadpool_info()` se detectó que el NumPy de este entorno usa Intel
MKL con 4 threads por proceso por defecto, y OpenMP con 8. Eso significa
que al lanzar `joblib.Parallel(n_jobs=8)`, cada uno de los 8 procesos
podía abrir además sus propios threads de BLAS, llegando en teoría a
muchos más threads que cores físicos hay en la máquina. La primera
medición que hicimos, una sola corrida, mostró una mejora clara al fijar
`threadpool_limits(1)` (de 1.857 s bajó a 1.465 s). Pero esta máquina es
una VM de WSL2 compartida con el host de Windows, el ruido entre
corridas es alto, y al repetir la medición varias veces en la parte (e),
usando la mediana de 3 repeticiones por configuración, esa mejora no se
sostuvo de forma tan clara. Dejamos el detalle completo, con la tabla
real de p y t, en la parte (e) más abajo, junto con la lección de fondo,
una sola corrida no alcanza para concluir que un efecto de rendimiento
es real. Igual mantuvimos el flag `--threads`/`-t` en las 3 versiones
(vía `common.build_argparser`), porque no medir y no poder controlar el
número de threads internos habría sido peor que medirlo y encontrar que
en esta máquina cambia menos de lo que pensábamos, además es justo lo
que pide la parte (i)
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

## Parte (c). Correctitud y reproducibilidad

Esta parte responde dos preguntas separadas sobre las tres versiones. Primero, si corremos lo mismo dos veces obtenemos lo mismo (reproducibilidad). Segundo, si las tres versiones calculan lo mismo aunque tomen caminos distintos (correctitud). El script `verify_correctness.py` deja esto medido con números reales, no solo explicado en palabras. Se corre así.

```
python verify_correctness.py --p 4 --B 48 --threads 1
```

### Reproducibilidad

Se corrió cada versión dos veces, con los mismos datos y las mismas semillas, y se compararon los arreglos de coeficientes resultantes con `np.array_equal`, que exige que sean idénticos bit a bit, no solo parecidos.

Las tres versiones dieron exactamente lo mismo en ambas corridas. Esto no es casualidad, viene de cómo quedó armado todo desde la parte (a). `generate_data.py` usa una semilla fija (42) para generar `X`, `y` y `beta_true`. `common.RESAMPLE_SEED` (123) fija qué índices caen en cada uno de los 48 resamples de `bs_sklearn.py` y `bs_numpy.py`. Y `bs_auto.py` fija su propio `random_state` (también 123) para el resampleo interno de `BaggingRegressor`. Con la misma entrada, las operaciones de punto flotante que hace cada versión (sumar, multiplicar, resolver un sistema lineal) siempre producen el mismo resultado, así que no hay forma de que dos corridas den algo distinto.

Esto es justamente lo que hace posible que cualquiera del grupo corra `generate_data.py` en su propio computador y obtenga los mismos datos, sin tener que compartir los archivos `.npy` (ya lo habíamos comentado antes en este README). Ahora queda claro que lo mismo aplica a los resultados completos del bootstrap, no solo a los datos de entrada.

### Correctitud entre bs_sklearn y bs_numpy

Estas dos versiones usan exactamente los mismos 48 resamples (las mismas filas de `X` en cada resample). La única diferencia entre ellas es el método numérico usado para ajustar la regresión. `bs_sklearn.py` resuelve por mínimos cuadrados vía `scipy.linalg.lstsq` (una factorización tipo SVD), mientras que `bs_numpy.py` arma y resuelve directamente las ecuaciones normales con `np.linalg.solve`.

Si el resampleo es el mismo, ambos métodos deberían llegar prácticamente al mismo lugar, y eso es lo que se midió. Comparando los 48 por 301 coeficientes que calcula cada versión, la diferencia absoluta más grande entre ambas fue muchísimo más chica que cualquier coeficiente real (del orden de una billonésima), y el promedio de esa diferencia fue todavía más chico. Son valores del tamaño de la precisión que permite un `double` (52 bits de fracción, como se vio en la clase 2 al hablar de punto flotante). No dan cero exacto porque cada método suma y multiplica los números en un orden distinto internamente, y la suma en punto flotante no es asociativa, exactamente el fenómeno que se vio en la clase de paralelismo a nivel de instrucciones con el ejemplo de sumar 1e20 más menos 1e20 más 1.0, donde el 1.0 se pierde según el orden en que se sume. Que la diferencia entre ambas versiones sea de ese tamaño tan chico, y no algo como 0.01 o 1.0, es justamente la evidencia de que las dos implementaciones son correctas y están calculando lo mismo.

### Correctitud estadística de las tres versiones

`bs_auto.py` no comparte resamples con las otras dos (usa el generador interno de `BaggingRegressor`, ya lo señalamos en la parte (b)), así que no tiene sentido compararla coeficiente a coeficiente. Lo que sí se puede comparar es si las tres describen la misma distribución bootstrap, viendo cuántos de los 301 coeficientes verdaderos (`beta_true`) caen dentro del intervalo de confianza 95% que arma cada versión, y qué tan ancho es ese intervalo en promedio.

Estos son los resultados con B=48 y los datos de `generate_data.py`.

`bs_numpy` cubrió 275 de 301 coeficientes, con un ancho promedio de intervalo de 0.03702.

`bs_sklearn` cubrió 275 de 301, ancho promedio 0.03702, prácticamente igual a `bs_numpy`, como era de esperar dado el punto anterior.

`bs_auto` cubrió 277 de 301, ancho promedio 0.03654.

Las tres cifras están cerca unas de otras, así que las tres versiones están estimando la misma distribución bootstrap aunque `bs_auto` haya llegado ahí con resamples distintos. Ninguna de las tres llega al 95% de cobertura nominal (que serían 286 de 301), y eso es esperable con B solo igual a 48. Un intervalo percentil necesita bastantes más resamples para que sus extremos (percentil 2.5 y percentil 97.5) queden bien estimados. No es un error de las implementaciones, es una limitación conocida del método con este tamaño de B, y también queda como material útil para comentar en el informe.

### Sobre usar solo lo visto en clase

Antes de seguir con la parte (d), dejamos apuntado de dónde sale cada explicación que dimos en las partes (a), (b) y (c), para que quede claro que todo se apoya en lo visto en clase y no en herramientas ajenas al curso.

La reproducibilidad de la parte (a) (semilla fija, mismos datos en cualquier computador) y la de esta parte (c) se explican con lo mismo, operaciones de punto flotante deterministas sobre la misma entrada, tal como se ve al hablar de precisión doble en la clase 2.

El worker de la parte (b) (cada uno de los p procesos que lanza `joblib.Parallel`) es exactamente la definición de worker que se dio en la clase 7, "un proceso del sistema operativo (Python multiprocessing)".

El hallazgo de oversubscription (demasiados threads de BLAS compitiendo por los mismos cores cuando ya hay varios procesos) se apoya en la idea de multicore de la clase 2 (varios núcleos físicos por chip) y en el concepto de worker de la clase 7. `threadpoolctl` solo la usamos como herramienta para medir y controlar algo que el curso ya explica en principio, no estamos trayendo un concepto nuevo de afuera.

La diferencia de velocidad entre invertir la matriz a mano y usar `np.linalg.solve` es un ejemplo directo de lo que dice la clase 2 sobre que la velocidad de un programa depende del algoritmo usado y no solo del hardware.

La pérdida de precisión al sumar en distinto orden, usada arriba para explicar por qué `bs_sklearn` y `bs_numpy` no dan exactamente lo mismo, es el mismo fenómeno de asociatividad de punto flotante que se vio en la clase de paralelismo a nivel de instrucciones.

## Parte (d). Como crea procesos el backend multiprocessing de joblib

Esta parte es teórica, no depende de tener las 3 versiones listas, así que cualquiera del grupo puede escribirla ya. Para no quedarnos solo con una explicación de memoria, escribimos `inspect_workers.py`, que hace que cada tarea (cada resample) devuelva el pid del proceso que la ejecutó, y con eso se puede ver con datos reales qué hace joblib por dentro.

Se corre así.

```
python inspect_workers.py --p 4 --B 8
```

### Con p igual a 1 no se crea ningún proceso nuevo

Cuando `n_jobs=1`, las 4 tareas de prueba corrieron todas con el mismo pid que el proceso principal. Joblib no crea ningún proceso nuevo en ese caso, simplemente ejecuta las tareas una por una dentro del mismo proceso que llamó a `Parallel`. Esto explica por qué en la parte (b) usamos p=1 como punto de comparación base, ahí no hay overhead de creación de procesos que restar.

### Con p mayor a 1, joblib arma un grupo de procesos y los reutiliza

Con `n_jobs=4` y 8 tareas, aparecieron solo 3 pids distintos entre las 8 tareas (esto varía entre corridas, otras veces salieron los 4). La cantidad de procesos worker nunca superó el p pedido, y cada proceso resolvió varias tareas seguidas, no se creó un proceso nuevo por cada una de las 48 resamples que usamos en la parte (b). Esto calza justo con el worker que se definió en la clase 7, "un proceso del sistema operativo". Joblib arma un grupo fijo de como máximo p procesos de esos al principio, y les va repartiendo las tareas a medida que van quedando libres, en vez de crear y destruir un proceso nuevo por cada resample, que sería mucho más caro.

### Cada proceso tiene su propia memoria

Lo importante de que sean procesos y no threads es que cada uno tiene su propio espacio de memoria, no comparten variables entre sí como sí lo hacen los threads dentro de un mismo proceso. En la clase 7 esto corresponde a la arquitectura MIMD con memoria distribuida, "cada procesador tiene su propia memoria, se comunican por mensajes". En nuestro caso los "procesadores" son procesos de Python corriendo en la misma máquina y no nodos de un cluster, pero la idea es la misma, no hay una memoria única compartida entre ellos como sí la hay entre threads que corren en cores distintos de un mismo chip. Eso lo vimos en la clase 5, ahí sí hace falta un protocolo de coherencia de caché como MESI porque los threads comparten memoria escribible. Acá no aplica, porque los procesos de joblib no comparten memoria escribible entre sí.

### El caso especial de arreglos grandes como X

Al inspeccionar qué tipo de objeto le llega a cada tarea, `X` (24 MB) llegó como `numpy.memmap` en vez de `numpy.ndarray` cuando p es mayor a 1, mientras que `y` (menos de 0.1 MB) siguió llegando como un `ndarray` normal. Esto quiere decir que joblib no copió los 24 MB de `X` para cada una de las tareas, en vez de eso los procesos worker leen el mismo arreglo mapeado desde un archivo temporal, sin duplicarlo por cada tarea. Si joblib copiara `X` completo cada vez que se lanza una tarea, con 48 resamples estaríamos moviendo más de 1 GB de memoria solo para repartir los datos, justo el tipo de costo que la clase 3 identifica como caro, mover datos tiene costo de ancho de banda y de latencia. Para `y`, como es chica, no vale la pena mapearla, se copia normal.

### La asignación de tareas no es reproducible, pero el resultado sí

Qué proceso ejecuta cuál tarea puede cambiar entre una corrida y otra, en nuestras pruebas a veces se usaron 3 procesos y otras veces 4 para el mismo p=4. Pero eso no rompe nada de lo que verificamos en la parte (c), porque cada tarea depende solo de su propia semilla y de leer `X` e `y`, nunca de una variable compartida que otra tarea pudiera estar modificando al mismo tiempo. Por eso el resultado final del bootstrap es siempre el mismo sin importar en qué orden ni en qué proceso se haya calculado cada resample.

## Parte (e). Oversubscription con threadpoolctl

Esta parte retoma algo que ya habíamos tocado de pasada en la parte (b), qué pasa cuando pedimos más hilos de trabajo de los que la máquina puede atender de verdad al mismo tiempo. Para medirlo con cuidado escribimos `oversubscription.py`, que corre así.

```
python oversubscription.py --B 48
```

### Cuántos threads abre NumPy por defecto

Con `threadpool_info()` se ve qué librería de álgebra lineal está detrás de NumPy en este entorno y cuántos threads usa por defecto. Acá aparecen dos, MKL para las operaciones de BLAS con 4 threads por defecto, y OpenMP con 8. Eso importa porque cada uno de esos threads es independiente del número de procesos p que le pidamos a `joblib.Parallel`. Si no hacemos nada, cada uno de los p procesos puede intentar abrir sus propios threads de MKL por su cuenta, sin que joblib se entere ni lo coordine.

### Qué es oversubscription en este contexto

Esta máquina tiene 8 cores lógicos (`os.cpu_count()`). Si lanzamos p procesos y cada uno usa t threads internos de BLAS, en total estamos pidiendo p multiplicado por t hilos de trabajo. Mientras ese número se mantenga en 8 o menos, cada hilo puede tener su propio core. Apenas ese número supera los 8 cores disponibles, varios hilos tienen que turnarse el mismo core, eso es oversubscription. Con p=8 y sin controlar t, en teoría podríamos llegar a 32 hilos compitiendo por 8 cores.

### Lo que medimos

Corrimos el ajuste de `bs_numpy.py` con distintas combinaciones de p y t, tomando la mediana de 3 repeticiones por combinación para no confiar en una sola corrida (esta máquina es una VM compartida con Windows, el ruido entre corridas es real, ya lo comentamos en la parte b). Estos son los tiempos, en segundos, sobre 48 resamples.

Con p=1 y t=1 (un solo proceso, un solo thread, cero paralelismo) el ajuste demoró 2.73 segundos, la configuración más lenta con diferencia.

Con p=1 y t=8 (un solo proceso, pero dejando que BLAS use los 8 cores por su cuenta) bajó a 0.93 segundos, casi tres veces más rápido, sin lanzar ni un proceso adicional.

Con p=8 y t=1 (8 procesos, cada uno restringido a un solo thread) quedó en 0.96 segundos, prácticamente empatado con la opción anterior, pero llegando ahí por un camino completamente distinto, repartiendo las 48 tareas entre 8 procesos en vez de acelerar cada tarea por dentro.

Con p=8 y t igual a 2, 4 u 8 (8 procesos, cada uno con varios threads internos, en teoría con 16, 32 y 64 hilos compitiendo por 8 cores) los tiempos quedaron entre 0.89 y 0.93 segundos, dentro del mismo rango que p=8 con t=1.

### Qué aprendimos de esto

Lo más claro y sólido de esta medición es que ir de cero paralelismo (p=1, t=1) a cualquiera de las otras opciones ayuda muchísimo, ahí está casi todo el rendimiento que se puede ganar en esta máquina para este problema.

Lo que no se sostuvo fue la diferencia grande que habíamos medido en la parte (b) entre p=8 sin controlar threads y p=8 con `threadpool_limits(1)`. Con una sola corrida esa diferencia se veía clara, pero al repetir la medición y tomar la mediana, todas las combinaciones con p=8 quedaron parecidas entre sí, estén o no oversubscritas según nuestro criterio de p por t contra los cores disponibles. La lección de fondo, la misma que ya habíamos aplicado en la parte (c) para la reproducibilidad, es que una sola corrida no alcanza para concluir que un efecto de rendimiento es real, sobre todo en una máquina compartida como esta.

Aun así, seguimos dejando el flag `--threads` en las 3 versiones y seguimos usando t=1 como valor por defecto cuando p se acerca a la cantidad de cores. No midió peor en ningún caso, y es la opción más simple de razonar, un core por proceso, así que la mantenemos como base para la parte (i), donde vamos a explorar la grilla completa de p y t con muchas más repeticiones, y ahí sí quedarnos con una conclusión firme sobre si conviene o no dejar que cada proceso use más de un thread interno cuando p ya está cerca del número de cores.

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
