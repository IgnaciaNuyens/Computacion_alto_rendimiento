# Tarea 1, IIC3533 Computación de Alto Rendimiento (2026-2)

Bootstrapping paralelo para regresión lineal con `joblib`. Grupo de 3 personas,
experimentos en al menos 2 computadores distintos. La entrega es el viernes 11
de septiembre 2026, a las 23:59.

## Estado actual

- [x] **(a) Generación de datos**, `generate_data.py` (hecho por Ignacia)
- [x] **(b) Tres implementaciones**, `bs_auto.py`, `bs_sklearn.py`, `bs_numpy.py` (más `common.py` con utilidades compartidas), ya iteradas y con la bitácora de mejoras documentada abajo. Falta pasar esto al informe en PDF.
- [x] **(c) Correctitud y reproducibilidad** hecha con `verify_correctness.py`, resultados y explicación abajo.
- [x] **(d) Cómo crea procesos el backend `multiprocessing` de joblib** hecha con `inspect_workers.py`, resultados y explicación abajo.
- [x] **(e) Oversubscription con `threadpoolctl`** hecha con `oversubscription.py`, resultados y explicación abajo.
- [x] **(f) Tiempos T(p) para p = 1..p_max, 3 versiones** hecha con `benchmark.py`, resultados y explicación abajo.
- [x] **(g) Speedup S(p) y eficiencia E(p)** hecha con `speedup_efficiency.py`, resultados y explicación abajo.
- [x] **(h) Overhead T_o(p)** hecha con `overhead.py`, resultados y explicación abajo.
- [x] **(i) Grid de (procesos p) x (threads t) con p·t ≤ p_max** hecha con `grid_pt.py`, resultados y explicación abajo.
- [ ] **(j) Comparación entre los 2 computadores**
- [ ] Informe final en PDF

## Cómo generar los datos

```
python generate_data.py
```

Genera `beta_true.npy`, `X.npy`, `y.npy` (N=10 000, k=300, semilla fija = 42).
No hace falta subir estos `.npy` al repo, porque la semilla está fija y
`numpy.random.default_rng` es determinista (no depende del hardware), así
que correr el script en cualquiera de los 2 computadores del grupo produce
exactamente los mismos datos. Por eso está en `.gitignore`. Si en algún
momento dudan de que dos máquinas dieron lo mismo, comparen con
`np.array_equal` o un hash (`np.save` más `sha256sum`).

## Entorno

```
conda create -n tarea1-hpc python=3.13 -y
conda activate tarea1-hpc
conda install numpy matplotlib joblib threadpoolctl scikit-learn -y
```

(El enunciado no lista `scikit-learn` en el comando de instalación, pero
`bs_auto.py` y `bs_sklearn.py` lo necesitan, de ahí salen `BaggingRegressor`
y `LinearRegression`.)

Instalen el mismo entorno en ambos computadores del grupo (misma versión de
Python/numpy si es posible, para que los tiempos sean comparables).

## Estructura del repo

```
generate_data.py         # parte a, genera X.npy, y.npy, beta_true.npy
common.py                # funciones compartidas por bs_auto/bs_sklearn/bs_numpy
bs_auto.py                # parte b, BaggingRegressor(n_jobs=p)
bs_sklearn.py              # parte b, joblib.Parallel + LinearRegression
bs_numpy.py                 # parte b, joblib.Parallel + ecuaciones normales
verify_correctness.py        # parte c
inspect_workers.py            # parte d
oversubscription.py            # parte e
benchmark.py                    # parte f, guarda tiempos en results/
speedup_efficiency.py            # parte g, lee el csv de benchmark.py
overhead.py                       # parte h, lee el csv de benchmark.py
grid_pt.py                         # parte i
results/                     # csv de tiempos, uno por computador (gitignored)
plots/                       # todavia no existe, figuras para el informe
informe/                     # todavia no existe, fuente del informe en pdf
```

## Parte (b). Iteración de las 3 implementaciones

El enunciado pide que comentemos las mejoras que hicimos, así que dejamos
apuntado acá lo que fuimos probando. Todo esto se corrió en el mismo
computador (8 cores lógicos), con los datos de `generate_data.py` y B=48.

**Invertir la matriz a mano versus `np.linalg.solve` en `bs_numpy.py`.**
La fórmula del enunciado es beta gorro igual a la inversa de X traspuesta
por X, todo multiplicado por X traspuesta por y. Al principio la
calculamos literal así, con `np.linalg.inv`. Después probamos
`np.linalg.solve`, que resuelve el mismo sistema de ecuaciones sin armar
la inversa completa, y nos quedó bastante más rápido.

| Variante | Tiempo (p=8, t=1, B=48) |
|---|---|
| `np.linalg.inv(XtX) @ Xty` (como sale literal en el enunciado) | 2.013 s |
| `np.linalg.solve(XtX, Xty)` (con la que nos quedamos) | 0.956 s, alrededor de 2 veces más rápido |

**Threads internos de BLAS.**
Con `threadpool_info()` vimos que el numpy de este entorno usa Intel MKL,
que por defecto abre 4 threads por proceso, y también OpenMP con 8. Eso
importa porque son threads que se abren aparte de los p procesos que le
pedimos a `joblib.Parallel`. Probamos limitarlos con `threadpool_limits`
y la primera corrida (una sola vez) dio una mejora bastante clara. Pero
esta máquina es una VM de WSL2 compartida con el host de Windows y tiene
harto ruido entre corridas, así que cuando lo volvimos a medir en la
parte (e) con varias repeticiones, esa mejora no se sostuvo tan clara.
Igual dejamos el flag `--threads` en las 3 versiones porque de todas
formas lo íbamos a necesitar para la parte (i).

**`copy_X=False`, una mejora que descartamos porque estaba mal.**
Probamos `LinearRegression(copy_X=False)` para que sklearn no haga una
copia interna extra de cada resample. En `bs_sklearn.py` funcionó bien,
cada resample ya es un array nuevo por el indexado `X[idx]`, así que no
hay problema en no copiarlo de nuevo. Pero en `bs_auto.py`, que usa
`BaggingRegressor`, los resultados salieron mal, la varianza de los
coeficientes bootstrap quedaba mucho más alta de lo que esperábamos y
además cambiaba entre p=1 y p=8. Investigando encontramos la razón,
`BaggingRegressor` no arma un array nuevo por estimador como hacíamos
nosotros a mano, reutiliza el mismo `X` y le pasa un `sample_weight` con
las repeticiones del bootstrap, así que con `copy_X=False` cada ajuste
modificaba ese mismo array compartido. Volvimos a dejarlo en el valor por
defecto (`copy_X=True`) en `bs_auto.py`. Nos pareció importante dejarlo
anotado porque muestra que iterar también es probar algo, notar que da
mal, y devolverse.

**Tiempos con la versión final de cada script, en este computador.**

| Versión | p=1, t=8 | p=8, t=1 |
|---|---|---|
| `bs_numpy` | 1.25 s | 2.15 s |
| `bs_sklearn` | 4.69 s | 7.01 s |
| `bs_auto` | 5.67 s | 11.55 s |

En esta máquina, usar un solo proceso con 8 threads internos salió más
rápido que usar 8 procesos con un solo thread, en las 3 versiones. Esto
lo volvemos a revisar con más cuidado en las partes (e) e (i). `bs_numpy`
es la más rápida de las tres porque resuelve directo las ecuaciones
normales, `bs_sklearn` ajusta con sklearn (que por dentro usa una
factorización más robusta pero más cara), y `bs_auto` es la más lenta
porque además tiene el overhead propio de `BaggingRegressor`.

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

`bs_numpy` cubrió 278 de 301 coeficientes, con un ancho promedio de intervalo de 0.03717.

`bs_sklearn` cubrió 278 de 301, ancho promedio 0.03717, prácticamente igual a `bs_numpy`, como era de esperar dado el punto anterior.

`bs_auto` cubrió 277 de 301, ancho promedio 0.03654.

Las tres cifras están cerca unas de otras, así que las tres versiones están estimando la misma distribución bootstrap aunque `bs_auto` haya llegado ahí con resamples distintos. Ninguna de las tres llega al 95% de cobertura nominal (que serían 286 de 301), y eso es esperable con B solo igual a 48. Un intervalo percentil necesita bastantes más resamples para que sus extremos (percentil 2.5 y percentil 97.5) queden bien estimados. No es un error de las implementaciones, es una limitación conocida del método con este tamaño de B, y también queda como material útil para comentar en el informe.

### Sobre usar solo lo visto en clase

Tratamos de explicar todo con lo que hemos visto en las clases, sin meter conceptos de afuera. La idea de worker como proceso del sistema operativo es de la clase 7, la de multicore y que la velocidad depende del algoritmo y no solo del hardware es de la clase 2, y la pérdida de precisión al sumar en distinto orden (por qué `bs_sklearn` y `bs_numpy` no dan exactamente lo mismo) es la misma idea de punto flotante que se vio en esa clase. `threadpoolctl` lo usamos solo como herramienta para medir algo que el curso ya explica en principio.

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

## Parte (f). Tiempos T(p) para p desde 1 hasta p_max, en las 3 versiones

Con las partes (b), (c), (d) y (e) ya resueltas, esta parte junta todo en un solo experimento, cuánto se demora cada versión al variar p desde 1 hasta p_max. Escribimos `benchmark.py` para esto, se corre así.

```
python benchmark.py --pmax 8 --B 48 --repeticiones 3
```

Usa threads=1 fijo, la opción que dejamos como base en la parte (e), y B=48 resamples, igual que en todas las partes anteriores. Aprendimos en la parte (e) que una sola corrida no alcanza en esta máquina compartida, así que `benchmark.py` repite cada combinación de versión y p tres veces y guarda las tres, no solo un promedio. Los resultados quedan en `results/benchmark_<nombre del computador>.csv`, listos para las partes (g), (h) y (j) sin tener que correr nada de nuevo, y como el nombre del archivo incluye el nombre del computador, cuando cada integrante lo corra en el suyo los archivos no se van a pisar entre sí.

Esta es la mediana de las 3 repeticiones para cada combinación, en segundos, en este computador (`p_max` igual a 8 cores).

| p | bs_numpy | bs_sklearn | bs_auto |
|---|---|---|---|
| 1 | 2.832 | 7.896 | 8.868 |
| 2 | 2.047 | 8.718 | 10.040 |
| 3 | 1.407 | 4.574 | 6.925 |
| 4 | 1.485 | 5.670 | 5.350 |
| 5 | 1.446 | 5.009 | 4.084 |
| 6 | 1.404 | 6.234 | 4.095 |
| 7 | 1.397 | 4.329 | 5.258 |
| 8 | 1.117 | 4.013 | 4.875 |

Algunas cosas para comentar en el informe, con estos números todavía crudos, antes de calcular speedup y eficiencia en la parte (g).

`bs_numpy` es la más rápida en todos los p, y baja de forma bastante pareja a medida que p crece, de 2.832 segundos con un solo proceso a 1.117 con 8, algo más de dos veces y media más rápido, lejos del ideal de ocho veces que pediría un speedup perfecto.

`bs_sklearn` y `bs_auto` no bajan de forma pareja, en p=2 las dos incluso empeoraron un poco respecto a p=1, y recién mejoran de forma clara desde p=3 en adelante. Esto es ruido de la máquina compartida (lo mismo que ya vimos en la parte e), pero también algo real, con solo 48 tareas para repartir, pasar de 1 a 2 procesos agrega el costo de crear un proceso nuevo (visto en la parte d) sin alcanzar a repartir tanto trabajo todavía, así que a veces el resultado no compensa de inmediato.

Ninguna de las tres versiones se acerca al speedup ideal de p, para eso ya tenemos la parte (g), que va a poner estos mismos números en la fórmula de speedup y eficiencia de la clase 7 y va a dejar más claro cuánto de esta diferencia es overhead real y cuánto es la parte que efectivamente sí se pudo paralelizar.

## Parte (g). Speedup S(p) y eficiencia E(p)

Esta parte no necesita correr nada nuevo, usa los mismos tiempos T(p) que ya quedaron guardados en la parte (f), en `results/benchmark_<nombre del computador>.csv`. Escribimos `speedup_efficiency.py`, que lee ese archivo y calcula, para cada versión, exactamente las fórmulas de la clase 7, Sp igual a T1 dividido por Tp, y Ep igual a Sp dividido por p. Se corre así.

```
python speedup_efficiency.py --hostname LAPTOP-DJ126R18
```

(cambien `LAPTOP-DJ126R18` por el nombre de su propio computador, que es el mismo que aparece en el nombre del archivo csv que generó `benchmark.py`).

Estos son los resultados en este computador, usando T(1) de cada versión como base.

### bs_numpy, T(1) es 2.832 segundos

| p | T(p) | S(p) | E(p) |
|---|---|---|---|
| 1 | 2.832 | 1.000 | 1.000 |
| 2 | 2.047 | 1.384 | 0.692 |
| 3 | 1.407 | 2.013 | 0.671 |
| 4 | 1.485 | 1.907 | 0.477 |
| 5 | 1.446 | 1.959 | 0.392 |
| 6 | 1.404 | 2.017 | 0.336 |
| 7 | 1.397 | 2.027 | 0.290 |
| 8 | 1.117 | 2.535 | 0.317 |

### bs_sklearn, T(1) es 7.896 segundos

| p | T(p) | S(p) | E(p) |
|---|---|---|---|
| 1 | 7.896 | 1.000 | 1.000 |
| 2 | 8.718 | 0.906 | 0.453 |
| 3 | 4.574 | 1.726 | 0.575 |
| 4 | 5.670 | 1.392 | 0.348 |
| 5 | 5.009 | 1.576 | 0.315 |
| 6 | 6.234 | 1.267 | 0.211 |
| 7 | 4.329 | 1.824 | 0.261 |
| 8 | 4.013 | 1.967 | 0.246 |

### bs_auto, T(1) es 8.868 segundos

| p | T(p) | S(p) | E(p) |
|---|---|---|---|
| 1 | 8.868 | 1.000 | 1.000 |
| 2 | 10.040 | 0.883 | 0.442 |
| 3 | 6.925 | 1.281 | 0.427 |
| 4 | 5.350 | 1.658 | 0.414 |
| 5 | 4.084 | 2.172 | 0.434 |
| 6 | 4.095 | 2.166 | 0.361 |
| 7 | 5.258 | 1.687 | 0.241 |
| 8 | 4.875 | 1.819 | 0.227 |

### Lo que dicen estos números

Ninguna de las tres versiones se acerca al speedup ideal, Sp igual a p con Ep igual a 1, que la clase 7 define como caso ideal. El caso más llamativo es p igual a 2, donde `bs_sklearn` y `bs_auto` tienen speedup por debajo de 1, o sea corren más lento con 2 procesos que con uno solo. Ya lo habíamos visto crudo en la parte (f), y acá queda numéricamente confirmado, con pocas tareas para repartir (48 resamples), el costo de crear un proceso adicional (que vimos en la parte d) puede superar por completo lo que se gana al paralelizar, sobre todo cuando cada tarea individual ya es rápida.

La eficiencia de las tres versiones cae de forma bastante consistente a medida que p crece, de 1.000 en p=1 a valores entre 0.23 y 0.32 en p=8. Esto es justo lo que la clase 7 anticipa para el caso práctico, Ep menor a 1 y decreciente, en contraste con el caso ideal.

`bs_numpy` es la que mejor se comporta en las dos métricas, mayor speedup en p=8 (2.535) y también mejor eficiencia en casi todos los p intermedios. Tiene sentido, cada una de sus 48 tareas es la más liviana de las tres versiones (resuelve directamente las ecuaciones normales, como vimos en la parte b), así que el trabajo útil por tarea es más chico en comparación con el overhead fijo de repartir esa tarea a un proceso, lo que en teoría debería jugar en contra suyo, pero en la práctica sale mejor parada que las otras dos de todas formas.

### Comparando con la ley de Amdahl

La clase 7 dice que el speedup máximo posible depende de qué tan grande sea la parte del programa que no se puede paralelizar. Ninguna de nuestras 48 tareas depende de otra, así que en teoría casi todo debería paralelizarse bien. Que el speedup real quede tan lejos de p, como se ve en las tablas de arriba, nos dice que lo que está limitando el resultado no es el algoritmo en sí, sino el overhead de crear y coordinar procesos que ya habíamos visto en la parte (d), más el ruido propio de esta máquina. La parte (h) mide ese overhead de forma directa.

## Parte (h). Overhead T sub o de p

Otra vez no corrimos nada nuevo, reutilizamos los mismos tiempos T(p) que ya quedaron guardados en la parte (f). Escribimos `overhead.py`, que calcula exactamente lo que definió la clase 7, T0 de p igual a p por Tp menos T1. T0 igual a 0 es el caso ideal, todo el tiempo que suman los p procesos es trabajo útil. T0 mayor a 0 es tiempo que se gastó en coordinación, esperas o trabajo repetido, y no en avanzar el cálculo. Se corre así.

```
python overhead.py --hostname LAPTOP-DJ126R18
```

Estos son los resultados en este computador. La columna T0 sobre T1 muestra el overhead como múltiplo del tiempo que tomaba correr todo con un solo proceso, para poder comparar entre versiones que parten de tiempos base muy distintos.

### bs_numpy, T1 es 2.832 segundos

| p | T(p) | p por T(p) | T0(p) | T0 sobre T1 |
|---|---|---|---|---|
| 1 | 2.832 | 2.832 | 0.000 | 0.000 |
| 2 | 2.047 | 4.093 | 1.261 | 0.445 |
| 3 | 1.407 | 4.221 | 1.389 | 0.491 |
| 4 | 1.485 | 5.940 | 3.108 | 1.098 |
| 5 | 1.446 | 7.229 | 4.397 | 1.553 |
| 6 | 1.404 | 8.424 | 5.592 | 1.975 |
| 7 | 1.397 | 9.778 | 6.946 | 2.453 |
| 8 | 1.117 | 8.937 | 6.105 | 2.156 |

### bs_sklearn, T1 es 7.896 segundos

| p | T(p) | p por T(p) | T0(p) | T0 sobre T1 |
|---|---|---|---|---|
| 1 | 7.896 | 7.896 | 0.000 | 0.000 |
| 2 | 8.718 | 17.436 | 9.541 | 1.208 |
| 3 | 4.574 | 13.722 | 5.826 | 0.738 |
| 4 | 5.670 | 22.681 | 14.786 | 1.873 |
| 5 | 5.009 | 25.046 | 17.151 | 2.172 |
| 6 | 6.234 | 37.402 | 29.507 | 3.737 |
| 7 | 4.329 | 30.301 | 22.405 | 2.838 |
| 8 | 4.013 | 32.105 | 24.210 | 3.066 |

### bs_auto, T1 es 8.868 segundos

| p | T(p) | p por T(p) | T0(p) | T0 sobre T1 |
|---|---|---|---|---|
| 1 | 8.868 | 8.868 | 0.000 | 0.000 |
| 2 | 10.040 | 20.081 | 11.212 | 1.264 |
| 3 | 6.925 | 20.776 | 11.908 | 1.343 |
| 4 | 5.350 | 21.401 | 12.532 | 1.413 |
| 5 | 4.084 | 20.418 | 11.550 | 1.302 |
| 6 | 4.095 | 24.570 | 15.702 | 1.771 |
| 7 | 5.258 | 36.806 | 27.938 | 3.150 |
| 8 | 4.875 | 39.000 | 30.132 | 3.398 |

### Lo que dicen estos números

En las tres versiones T0 crece con p, y crece más rápido que T1. En p=8, el overhead ya es entre 2 y 3.4 veces el tiempo que tomaba correr todo con un solo proceso, según la versión. Dicho de otra forma, de todo el tiempo de cómputo que sumaron los 8 procesos juntos, más de dos tercios en el peor caso se fue en coordinación y esperas, no en avanzar el ajuste del bootstrap.

`bs_numpy` tiene el overhead relativo más bajo en casi todos los p, lo que calza con que también fue la versión con mejor eficiencia en la parte (g). Tiene sentido, S(p), E(p) y T0(p) están midiendo lo mismo con fórmulas distintas de la clase 7, así que si una versión sale mejor parada en una, debería salir mejor parada en las otras también.

También se nota que T0 no crece de forma perfectamente pareja, por ejemplo en `bs_numpy` el overhead en p=8 es un poco más chico que en p=7. Es el mismo ruido de la máquina compartida que venimos comentando desde la parte (e), y otra razón para tomar estos números como una tendencia general y no como valores exactos.

Esta fórmula de T0 junta en un solo número dos cosas que en partes anteriores sí separamos, el costo real de crear y coordinar procesos (parte d) y el ruido propio de esta máquina (parte e). La parte (i), variando threads además de procesos, nos va a mostrar si parte de este overhead se puede reducir eligiendo mejor la combinación de p y t.

## Parte (i). Grilla de procesos p y threads t con p por t hasta p_max

Las partes (e) y (h) dejaron una pregunta abierta, para una cantidad fija de hilos de trabajo, conviene más dejar que cada proceso use varios threads internos de BLAS, o conviene repartir en más procesos con un solo thread cada uno. Esta parte responde eso de forma directa, probando todas las combinaciones de p y t que cumplen p por t menor o igual a p_max (8 en esta máquina), no solo las combinaciones sueltas que ya habíamos probado antes. Escribimos `grid_pt.py` para esto, se corre así.

```
python grid_pt.py --pmax 8 --B 48 --repeticiones 3
```

Por defecto corre sobre `bs_numpy`, la versión con menor overhead según las partes (f), (g) y (h), aunque el script también acepta `--version bs_sklearn` o `--version bs_auto` para correr la misma grilla sobre las otras dos.

Estos son los 20 pares de p y t posibles con p por t hasta 8, con la mediana de 3 repeticiones cada uno, para `bs_numpy`.

| p | t | p por t | tiempo (s) |
|---|---|---|---|
| 1 | 1 | 1 | 2.408 |
| 1 | 2 | 2 | 1.188 |
| 1 | 3 | 3 | 1.036 |
| 1 | 4 | 4 | 1.004 |
| 1 | 5 | 5 | 0.994 |
| 1 | 6 | 6 | 1.227 |
| 1 | 7 | 7 | 1.582 |
| 1 | 8 | 8 | 1.478 |
| 2 | 1 | 2 | 3.243 |
| 2 | 2 | 4 | 1.838 |
| 2 | 3 | 6 | 2.574 |
| 2 | 4 | 8 | 3.006 |
| 3 | 1 | 3 | 1.751 |
| 3 | 2 | 6 | 1.558 |
| 4 | 1 | 4 | 1.607 |
| 4 | 2 | 8 | 1.588 |
| 5 | 1 | 5 | 1.384 |
| 6 | 1 | 6 | 1.406 |
| 7 | 1 | 7 | 1.656 |
| 8 | 1 | 8 | 1.492 |

### La mejor combinación no fue la que esperábamos

La combinación más rápida de las 20 fue p igual a 1 con t igual a 5, en 0.994 segundos, un solo proceso dejando que BLAS use 5 threads internos. No fue ninguna de las combinaciones con varios procesos que veníamos usando desde la parte (b). Mirando solo la fila de p igual a 1, el tiempo baja rápido desde t igual a 1 (2.408 segundos) hasta un piso entre t igual a 3 y t igual a 5 (alrededor de 1 segundo), y después vuelve a subir un poco en t igual a 6, 7 y 8. O sea, ni siquiera dentro de un solo proceso conviene usar todos los threads disponibles, hay un punto medio.

Con p mayor a 1 los tiempos nunca bajaron de 1.3 segundos, y el peor resultado de toda la grilla fue justamente p igual a 2 con t igual a 1 (3.243 segundos), más lento incluso que p igual a 1 con t igual a 1.

### Por qué pasa esto

Esto calza con todo lo que veníamos midiendo. En la parte (d) vimos que cada proceso worker tiene un costo real de creación y coordinación, y que joblib arma un grupo de procesos que se reutiliza, no procesos gratis. Un thread interno de BLAS no paga ese mismo costo, corre dentro del mismo proceso que ya está corriendo, comparte su memoria sin necesidad de mensajes ni de mapear archivos como vimos que hace joblib con arreglos grandes en la parte (d). Para las 48 tareas de nuestro bootstrap, cada una relativamente liviana (resolver un sistema de 301 por 301, no una tarea larga), el costo de abrir un proceso nuevo pesa más que el costo de abrir un thread nuevo, así que en esta máquina, para este problema, escalar con threads le gana a escalar con procesos.

Esto no quiere decir que la parte (b) haya estado mal al elegir joblib con procesos como esquema principal, el enunciado pide específicamente paralelismo con procesos, y además un problema más grande, con resamples más caros o con B mucho mayor a 48, podría cambiar esta conclusión, porque ahí el costo fijo de crear procesos pesaría menos frente al trabajo real de cada tarea. Lo que esta parte (i) deja claro es que esa suposición hay que medirla, no darla por sentada, y que la mejor combinación de p y t depende del tamaño del problema, no es siempre la misma.

## Qué es del enunciado y qué agregamos nosotros

Dejamos esto anotado para que quede claro entre nosotros tres, y por si en
algún momento nos piden explicar por qué el código tiene tantas partes.
Las 10 partes con letra, de la (a) a la (j), son las que pide el enunciado
tal cual, cada una con su propio archivo. Lo que sí fue iniciativa nuestra
es cómo las resolvimos adentro, por ejemplo medir cada tiempo 3 veces y
usar la mediana en vez de una sola corrida, porque en las primeras pruebas
nos dimos cuenta que esta máquina compartida da resultados bastante
distintos de una corrida a otra (queda contado en la parte b y en la
parte e). También armamos `common.py` para no repetir el mismo código de
cargar los datos en los 3 scripts de la parte (b). Nada de esto agrega
partes que el enunciado no pida, es simplemente cómo decidimos ordenar el
código para poder correrlo varias veces sin reescribir cosas.

## Cómo va la división del trabajo

Hasta ahora Ignacia avanzó sola con las partes (a) a la (i), documentando
cada una acá en el README a medida que las iba terminando. Falta que el
grupo revise el código junto para que los 3 podamos explicar cualquier
parte si nos preguntan, y falta la parte (j), que necesita correr
`benchmark.py` en un segundo computador para poder comparar. Eso lo puede
hacer cualquiera de los otros dos integrantes, siguiendo las instrucciones
de la parte (f) más abajo. Después de eso queda armar el informe en PDF
entre los 3, juntando lo que ya está escrito acá.

## Flujo de trabajo con git

```
git clone https://github.com/IgnaciaNuyens/Computacion_alto_rendimiento.git
git add archivo.py
git commit -m "mensaje corto"
git pull --rebase origin main
git push origin main
```

Como somos pocos y el código son scripts chicos, no hace falta un flujo
muy formal. Basta con avisar por el grupo antes de tocar un archivo que
otra persona está editando, y comentar el push en el chat.

## Dónde compartimos el código

El repo está en GitHub, es privado y ya tiene agregados a los 3
integrantes como colaboradores.

```
https://github.com/IgnaciaNuyens/Computacion_alto_rendimiento.git
```
