# Posesión por frame · More than a Hack 2027 · Fase 1

Documento técnico

> Equipo Veyon · Repositorio https://github.com/VEYON-DEV/more-than-a-hack-2027-veyon · Archivo `Veyon.csv`

---

## 1. Resumen

Asignamos una etiqueta a cada uno de los 199,319 frames del video táctico del partido entre Barça (H) y Athletic (A). Las etiquetas posibles son `H`, `A`, `D` (pelota muerta) y `X`.

```
tracking + video + broadcast
  → seguimiento de la pelota con Kalman y verificación de equipos por color
  → 77 variables de comportamiento colectivo por frame
  → 12 redes temporales con varias definiciones de posesión
  → decodificación con Viterbi y reglas
  → CSV
```

- Las redes se entrenan exclusivamente con otros partidos, procedentes de IDSSE (Bundesliga, CC-BY 4.0). Modificamos esos datos para reproducir las características de nuestro tracking, incluida la forma en que se registran las paradas.
- El entrenamiento, la validación y el ajuste se hicieron sin etiquetas humanas del partido evaluado.
- Todas las decisiones se tomaron mediante validación cruzada en los 7 partidos de IDSSE. En cada ronda dejamos un partido fuera del entrenamiento y lo medimos con el evaluador oficial.

## 2. Motivos del diseño

- La métrica `spellF1` premia la detección de posesiones completas. Las predicciones por frame pueden alternar rápidamente entre estados. Para formar tramos coherentes, las procesamos con Viterbi y exigimos una duración mínima.
- En algunas partes del partido, la pelota falta o su posición es incorrecta. La conducta de los 22 jugadores ayuda a identificar la posesión aun cuando no vemos la pelota. Entre otras señales, observamos quién presiona, quién avanza y quién corre hacia ella.
- Como no teníamos etiquetas del partido, evaluamos el diseño en 7 partidos completos con respuestas conocidas. La mayor mejora del proyecto vino de medir las diferencias entre nuestro tracking y el de entrenamiento y reproducirlas en los datos de entrenamiento (sección 4).

## 3. Datos usados y descartados

| Fuente | Aplicación | Estado |
|---|---|---|
| ATD táctico (jugadores, equipos y pelota) | Posición y equipo de cada jugador, más la posición de la pelota cuando está disponible. | Usado tras filtrar tracks congelados y saltos imposibles. |
| Video táctico | Detección de candidatos a pelota y verificación del equipo de cada track por color de camiseta. | Usado. |
| Homografía táctica | Conversión entre píxeles y metros, ubicación de las líneas del campo y cálculo del tamaño esperado de la pelota. | Usada. |
| ATD del broadcast | Se alinea con el táctico mediante las posiciones de los jugadores. Los desfases son de 4014 y 26920 frames, respectivamente, para cada tiempo. Aporta la pelota en el 2.9 % de los frames. | Usado solo para la pelota. |
| Smart Tagging | Aporta dos reglas seguras de pelota muerta. La primera cubre el intervalo entre un gol y el saque del medio. La segunda cubre el periodo anterior al saque inicial del 2.º tiempo. También se usa como comprobación independiente, fuera del modelo. | Uso mínimo. |
| Video y homografía del broadcast | El costo era alto y el beneficio, bajo. | Descartados. |
| YOLOv8s preentrenado (COCO, clase «sports ball») | En este plano, la pelota mide 5–10 px. La tasa de aciertos fue de 32–46 % y la de errores, de 13–41 %. | Descartado. |
| Audio del video (silbato) | La transformada de Fourier permite detectar el silbato en la banda de 3.85–4.3 kHz. Como IDSSE carece de audio, no pudimos medir el efecto de esta señal. | No incluido. |
| Audio de los comentaristas | Podría considerarse «dato producido por humanos sobre el partido». | No usado. |

### Datos externos de entrenamiento

- Bassek, M., Rein, R., Weber, H., Memmert, D. *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195 (2025). [Artículo de IDSSE](https://doi.org/10.1038/s41597-025-04505-y)
  - Los datos proceden de la Deutsche Fußball Liga (DFL) y tienen licencia CC-BY 4.0.
  - La descarga es automática desde el espejo público `pysport/idsse-data` en Hugging Face.
- El conjunto contiene 7 partidos completos de Bundesliga y 2. Bundesliga de la temporada 2022/23. Incluye tracking a 25 Hz y los campos `BallPossession` y `BallStatus` en cada frame.
- Las etiquetas proceden del proveedor oficial, DFL / Sportec Solutions.
- En total son ~1 millón de frames (7 × ~143,000), todos de partidos distintos del Barça–Athletic.
- Del Barça–Athletic calculamos automáticamente dos estadísticas sin usar etiquetas humanas.
  - Duración de los huecos en el seguimiento de la pelota con juego vivo y con pelota muerta. Para determinar el estado usamos la salida de una versión anterior de nuestra solución.
  - Frecuencia con que la pelota aparece a los pies de un jugador durante las paradas.

## 4. Tratamiento del tracking faltante y ruidoso

1. **Pelota.** Combinamos tres fuentes.
   - Candidatos del video. Buscamos manchas blancas, redondas y aisladas, que estén fuera de las líneas y los puntos pintados del campo y tengan el tamaño esperado según la homografía.
   - Posiciones de la pelota en el ATD táctico.
   - Posiciones de la pelota en el ATD del broadcast.

   Un filtro de Kalman combina las mediciones, estima posición y velocidad en coordenadas métricas y aplica suavizado RTS. Selecciona las mediciones según la distancia de Mahalanobis y la calidad del candidato. Cubre huecos de hasta 0.5 s. En huecos de mayor duración, la pelota queda como desconocida. La cobertura alcanza el 71.6 % de los frames calificados, frente al 59 % con el ATD solo, y llega a ~98 % con juego vivo.

2. **Jugadores.** Eliminamos los tracks congelados, definidos como aquellos sin movimiento durante ≥ 1 s, y descartamos las velocidades por encima de 12 m/s.

3. **Equipos.** Determinamos el equipo de cada track mediante una votación por color de camiseta. El resultado coincide con el ATD en el 96.5 % de los tramos. Corregimos el equipo en el 2.5 %.

4. **Adaptación de IDSSE al tracking del reto (versión «v3»).** Usamos 3 semillas e incluimos también una versión de cada partido con los equipos espejados.
   - **Huecos de la pelota según el estado.** En nuestro tracking conocemos la posición de la pelota el 98 % del tiempo con juego vivo y el 43 % con pelota muerta. El ensuciado inicial dejaba la pelota visible ≈62 % del tiempo en cualquier estado, así que su desaparición no servía como señal de pelota muerta. Ahora sorteamos la duración de los huecos según lo medido en cada estado.
   - **Suavizado de los jugadores.** Los movimientos de nuestro tracking son más suaves. El 18 % de los jugador-frames corresponde a jugadores casi quietos, frente al 11–13 % en IDSSE. Reproducimos este efecto con una media móvil de 1 s para cada jugador.
   - **Otros errores.** Simulamos tramos en los que se pierde a un jugador, tramos con el equipo cambiado y casos en los que la pelota queda pegada unos frames a los pies de un jugador.

5. **Simulación de paradas.** Con pelota muerta, nuestro tracking registra la pelota a los pies o en las manos de un jugador con más frecuencia que el IDSSE adaptado.
   - 9.6 % frente a 2 % del tiempo en que todos caminan.
   - 39 % frente a 4 % antes de un saque de arco.

   Esto ocurre cuando se levanta la pelota para un lateral, cuando el arquero la acomoda o cuando alguien se la devuelve al jugador que va a sacar. Si entrenamos sin reproducir estas situaciones, el modelo las interpreta como juego vivo y corta las paradas. La mediana de nuestras paradas predichas era de 11.9 s, frente a unos 20 s en IDSSE.

   Para el entrenamiento añadimos una copia de IDSSE v3 con dos modificaciones. En la mitad de las paradas, la pelota queda a los pies del jugador que va a sacar. En el 30 % de las paradas largas, se simula una devolución.

## 5. Reglas de posesión implementadas

- **Estados.** El modelo trabaja con H, A y D. Una posesión es el tramo máximo de un equipo mientras la pelota está viva.
- **Control.** En las etiquetas de entrenamiento de IDSSE, un cambio de posesión se registra cuando el nuevo equipo conserva la pelota durante al menos 1.5 s. Si no alcanza ese tiempo, la posesión sigue siendo del equipo anterior. Esto cubre los toques sin control, los desvíos y los duelos aéreos.
  - El umbral aproxima la definición del desafío, «un jugador mantiene la pelota unos segundos».
  - El modelo es robusto con umbrales entre 1 y 2 s.
- **Definiciones de control en paralelo.** Cada red aprende una cabeza por definición.
  - Etiquetas originales, por toques.
  - Tiempos mínimos de control de 0.5, 1, 1.5 y 3 s.
  - Tres variantes de control por contacto, calculadas con el tracking limpio de IDSSE.

  La entrega toma la salida de la cabeza de 1.5 s. Las otras definiciones son tareas auxiliares para reducir la dependencia del modelo de una sola regla de control.

- **Pelota muerta.** El modelo aprende a identificarla a partir de señales como la pelota fuera de las líneas, su desaparición, el ritmo de los jugadores y la pelota quieta o a los pies de un jugador durante una parada. También aplicamos las dos reglas seguras de Smart Tagging.
- **Reanudación.** Al terminar una parada, la posesión corresponde claramente al equipo que ejecuta el saque.

## 6. Modelo final

- **Variables por frame.** Usamos 77 variables de base y sus promedios en ventanas de ±5 y ±10 s.
  - Distancias entre la pelota y cada equipo.
  - Control de la pelota, definido como tenerla a menos de 1 m de un jugador y que se mueva con él.
  - Velocidad de acercamiento de cada equipo y cuál llega primero.
  - Presión sobre el portador y jugadores que quedan detrás de la pelota.
  - Avance y compactación de cada equipo.
  - Dirección de la pelota en relación con el ataque, pelota fuera del campo, ritmo de los jugadores y huecos en el seguimiento de la pelota.
- **12 redes convolucionales temporales con varias definiciones de posesión.**
  - **Arquitectura.** 6 bloques dilatados con saltos de 1…32, un contexto de ≈±10 s y 96 canales.
  - **Salidas.** Una cabeza H/A/D para cada definición, además de cabezas para cambio H↔A, saque y pelota muerta.
  - **Entrenamiento.** 1500 pasos con promedio móvil de los pesos (EMA 0.995). Usamos IDSSE v3 y su copia con paradas simuladas.
  - **Ensamble.** Las 12 redes solo se distinguen por la semilla. Cada una usa equipos espejados al entrenar y al predecir. En la predicción, devolvemos la salida espejada a su orientación original, la promediamos con la salida de la vista normal y finalmente promediamos los resultados de las 12 redes.
- **Decodificación.**
  - Aplicamos Viterbi con las transiciones medidas en IDSSE, temperatura 0.4, factor ×2 para las transiciones H↔A y ×0.6 para vivo↔muerto.
  - La probabilidad de cambio H↔A que predice la red ajusta las transiciones en cada frame, con un peso de 1.5.
  - Exigimos una duración mínima de posesión de 1.5 s.
  - Aplicamos las reglas seguras de Smart Tagging.

## 7. Validación

Cada cifra es el promedio de 7 rondas. En cada una entrenamos con 6 partidos de IDSSE y medimos el resultado en el séptimo, que se mantiene fuera del entrenamiento, con el evaluador oficial. Solo comparamos puntajes dentro del mismo «examen», con la misma versión de las modificaciones aplicadas a los datos de prueba.

| Paso | Examen | Antes → después | Partidos en que mejora |
|---|---|---|---|
| Reimplementación del baseline oficial (jugador más cercano + mayoría en ±2 s) | v1 | 0.202 | — |
| Gradient boosting + Markov | v1 | 0.672 | — |
| Huecos de pelota en función del estado del juego (v1 → v2) | v2 | 0.746 → 0.776 | 6/7 |
| Suavizado de jugadores para reproducir nuestro tracking (v2 → v3) | v3 | 0.780 → 0.797 | 6/7 |
| Ensamble de 4 redes con una sola definición (modelo base) | v3 | 0.794 | — |
| Modelo final con varias definiciones, paradas simuladas y 12 redes | v3 + paradas como las nuestras | 0.750 → 0.789 | 7/7 |
| Mismo modelo final | Variante no vista durante el entrenamiento | 0.748 → 0.788 | 7/7 |
| Mismo modelo final | v3 sin paradas simuladas | 0.790 → 0.796 | 5/7 |

Las últimas filas corresponden a comparaciones pareadas con el modelo base (3 semillas). El modelo final mejora con claridad cuando aparecen paradas como las de nuestro partido y mantiene el rendimiento cuando ese fenómeno está ausente.

En el examen v3 se detecta el 98 % de las posesiones de más de 20 s, el 82 % de las de 5–10 s y el 51 % de las de menos de 3 s. Esas posesiones cortas representan el 22 % del total. Con equipos perfectos, el puntaje llega a 0.97. Casi todo el error se concentra en las posesiones cortas.

En el Barça–Athletic comparamos el modelo base con el final sin disponer de una referencia para el partido completo.

- Pelota muerta de 45.0 % a 47.2 %.
- Mediana de duración de las paradas de 11.9 s a 14.7 s.
- Pelota muerta cuando todos caminan de 93.8 % a 97.4 %, frente al 96.8 % en IDSSE real.
- Finales de parada seguidos de un saque de 86 % a 90 %.
- Coincidencia con Smart Tagging en el centro de las fases con equipo de 90.5 % a 90.9 %.
- Revisamos en el video cuatro tramos nuevos marcados como pelota muerta. Los cuatro eran paradas reales (un cambio, un tiro libre, un saque de arco).

**Ventana pública de 5 minutos.** Solo usamos esa ventana para comprobar que el archivo estuviera bien armado y para entender la definición de posesión, como indica CHALLENGE.md. Encontramos 17 posesiones predichas frente a 17 reales, con desfase cero. También miramos dos paradas de la ventana como comprobación durante el diagnóstico de las paradas. La simulación se calibró con estadísticas automáticas y la decisión se tomó en el banco de pruebas de IDSSE, sin ajustar parámetros con las etiquetas de la ventana pública.

### Ideas probadas y descartadas

| Idea | Resultado | Decisión |
|---|---|---|
| Modelo de grafo con atención entre jugadores y una red temporal | Mejora de +0.007 en el banco. En nuestro partido predice 40 % de pelota muerta, frente al 46 % de las otras redes. | Descartado porque la mejora no se traslada a nuestro partido. |
| Variables de pelota parada (córner, banda, área) | 0.725 frente a 0.730 sin esas variables. | Descartadas. |
| Red temporal con contexto de ±20 s | 0.780 frente a 0.783. | Descartada. |
| Autoentrenamiento con pseudo-etiquetas del partido de prueba | 0.776 frente a 0.783. Las pseudo-etiquetas solo cubren los casos fáciles. | Descartado. |
| Incorporar más partidos de entrenamiento | Curva de aprendizaje plana, con 3 → 0.761, 4 → 0.770, 5 → 0.781 y 6 → 0.777. | Se mantuvo el conjunto de partidos. |
| Aumentar el peso de la pérdida cerca de los cambios de posesión | 0.790 / 0.782 / 0.778 frente a 0.798. | Descartado. |
| Usar un detector de toques para añadir variables | 0.788 frente a 0.798. | Descartado. |
| Modelo de segundo nivel (stacking) | 0.783 frente a 0.794. | Descartado. |
| Reproducir en IDSSE los errores reales de detección de nuestra pelota en video y los saltos de identidad de los jugadores | Empate en el examen corregido. | Descartado. |
| Red por etapas (MS-TCN) / objetivos de contacto real | +0.002 / +0.005, dentro de la variación entre semillas (~0.02). | Descartados. |
| Usar la definición por contacto en la entrega en lugar de la de 1.5 s | +0.05 si la definición oficial es por contacto y −0.08 si es por duración. | Descartado. |

## 8. Limitaciones

- IDSSE corresponde a otra liga y a otro sistema de tracking. La definición oficial de posesión se deduce de la información disponible.
- Las etiquetas oficiales incluyen posesiones de menos de 1.5 s que nuestras etiquetas de entrenamiento no contienen. Por eso, el banco de pruebas sobreestima ligeramente la nota real.
- El error principal sigue estando en las posesiones muy cortas, como los vuelos de la pelota tras un único toque. Para resolverlas haría falta distinguir cada toque con más precisión de la que permite el tracking disponible.
- La homografía del plano del suelo proyecta mal la pelota cuando está en el aire.
- Las modificaciones de IDSSE reproducen las diferencias que logramos medir. Pueden existir otras que no detectamos.
- Con más tiempo, entrenaríamos un detector de pelota sobre el video usando la pelota del ATD como etiqueta automática. Para la decodificación, probaríamos un modelo de duraciones semi-Markov.

## 9. Reproducción

```bash
python3 -m venv .venv       && .venv/bin/pip install -r requirements-solution.txt
python3 -m venv .venv-torch && .venv-torch/bin/pip install -r requirements-torch.txt
# datos del partido en data/ (ver data/README.md); IDSSE se descarga solo
.venv/bin/python solution/run_all.py --out evaluation/results/Veyon.csv
```

- Probamos la solución con Python 3.9.6 en macOS sobre Apple Silicon. Las redes se pueden ejecutar en GPU (MPS o CUDA) o en CPU.
- Usamos los pesos guardados en `solution/models/` porque el entrenamiento en GPU no es determinista bit a bit. Con esos pesos, el archivo generado coincide byte a byte con el que entregamos.
- La opción `--retrain` vuelve a entrenar las 12 redes desde cero con los mismos parámetros.
- Desde cero, el proceso tarda unos 16 minutos en un MacBook con Apple Silicon. La mayor parte del tiempo se dedica a leer el video.
