# More than a Hack 2027 · Fase 1 · Equipo Veyon

Este fork del repositorio oficial del reto contiene nuestra solución para estimar la posesión en cada frame del Barça–Athletic. Nuestro código está en `solution/` y en `explore/entity_colors.py`. El resto del repositorio se conserva tal como venía, incluidas las bases del reto en `CHALLENGE.md`, `RULES.md` y `SUBMISSION.md`.

El documento técnico detalla qué datos usamos y cuáles descartamos, cómo definimos la posesión, qué experimentos hicimos y qué limitaciones encontramos. [Documento técnico (docs/Veyon.md)](docs/Veyon.md)

## Método

Entrenamos el modelo con otros partidos, sin utilizar etiquetas del Barça–Athletic. Del partido evaluado solo usamos el tracking, el video, la homografía, el Smart Tagging y estadísticas calculadas automáticamente a partir de ellos.

### 1. Reconstrucción de la trayectoria de la pelota

El ATD táctico registra la pelota en el 59 % de los frames que se califican. Para cubrir parte de los huecos, añadimos dos fuentes:

- **Video táctico.** El detector busca manchas blancas en el espacio de color Lab. Usa la homografía para estimar el tamaño esperado de la pelota y selecciona candidatos redondos y aislados, fuera de las líneas y los puntos pintados del campo.
- **ATD del broadcast.** Como su tracking viene desfasado respecto del táctico, los alineamos comparando las posiciones de los jugadores. El desfase es de 4014 frames en el primer tiempo y de 26920 en el segundo.

Combinamos las tres fuentes con un filtro de Kalman que estima la posición y la velocidad en coordenadas métricas y aplica suavizado RTS. El filtro acepta mediciones según su distancia de Mahalanobis y la calidad del candidato, y cubre huecos de hasta 0.5 s. Así obtenemos la posición de la pelota en el 71.6 % de los frames calificados y en aproximadamente el 98 % de los frames con juego vivo.

### 2. Jugadores y equipos

Filtramos los tracks congelados y las velocidades imposibles. Para cada track, usamos la homografía para recortar la camiseta en el video y asignamos el equipo mediante una votación por color. La asignación coincide con el ATD en el 96.5 % de los tramos. Corregimos el equipo en el 2.5 %.

### 3. Variables por frame

Calculamos 77 valores por frame y sus promedios en ventanas de ±5 y ±10 s. Las variables incluyen:

- Distancia de la pelota a cada equipo y quién la controla. Para considerar que un jugador tiene el control, debe estar a menos de 1 m de la pelota y moverse con ella.
- Velocidad a la que se acerca cada equipo y cuál llegaría primero a la pelota.
- Presión sobre el portador y jugadores que quedan detrás de la pelota.
- Avance y compactación de cada equipo.
- Dirección de la pelota con respecto al sentido del ataque.
- Pelota fuera del campo, pelota quieta, huecos en su seguimiento y ritmo de los jugadores.

### 4. Datos de entrenamiento

Usamos IDSSE, un conjunto de siete partidos de la Bundesliga con tracking a 25 Hz y etiquetas de la DFL para la posesión y el estado de la pelota en cada frame. Como el tracking de IDSSE es más limpio que el del reto, introdujimos algunas de las diferencias que medimos en el Barça–Athletic. Cada uno de los siguientes ajustes mejoró el resultado en validación:

| Diferencia medida | Cómo la reproducimos en IDSSE | Mejora |
|---|---|---:|
| La pelota se pierde el 2 % del tiempo con juego vivo y el 57 % con pelota muerta. | Introducimos huecos al azar según el estado del juego. | +0.030 |
| El movimiento de los jugadores es más suave, con un 18 % casi quietos frente al 11–13 % en IDSSE. | Aplicamos una media móvil de 1 s a cada jugador. | +0.017 |
| Durante las paradas, la pelota aparece a los pies de un jugador con mayor frecuencia. Antes de un saque de arco, ocurre en el 39 % de los casos frente al 4 % en IDSSE. | En la mitad de las paradas dejamos la pelota con quien va a sacar. En el 30 % de las paradas largas, simulamos un pase de devolución. | +0.039 |

También simulamos tramos con jugadores perdidos o con el equipo mal asignado. Durante el entrenamiento usamos versiones espejadas con los equipos intercambiados para que el modelo no dependa de quién juega de local.

### 5. Modelo

Usamos 12 redes convolucionales temporales. Cada una tiene seis bloques con dilataciones de 1 a 32, 96 canales y un contexto de aproximadamente ±10 s. Cada red predice H/A/D en paralelo para la posesión por toque, con tiempos mínimos de control de 0.5, 1, 1.5 y 3 s, y para tres variantes por contacto. También estima la probabilidad de cambio de posesión, de saque y de pelota muerta.

La entrega usa la definición de 1.5 s. Las demás funcionan como tareas auxiliares. Las 12 redes solo difieren en la semilla. Para la predicción final, promediamos las salidas de todas las redes en la vista original y en la espejada.

### 6. Decodificación

Aplicamos Viterbi con las transiciones medidas en IDSSE. La probabilidad de cambio que estima la red ajusta esas transiciones en cada frame. Exigimos que cada posesión dure al menos 1.5 s. Según dos reglas de Smart Tagging, marcamos la pelota como muerta desde un gol hasta el saque siguiente y antes del saque inicial del segundo tiempo.

### 7. Validación

Hacemos validación cruzada con los siete partidos de IDSSE. En cada ronda, entrenamos con seis y evaluamos con el partido restante. Usamos el evaluador oficial y las mismas particiones para comparar todos los modelos.

| Modelo | spellF1 |
|---|---:|
| Cuatro redes con una sola definición de posesión y sin simulación de paradas | 0.750 |
| Doce redes con varias definiciones de posesión y simulación de paradas (entrega) | 0.789 |

Como referencia, el baseline oficial reimplementado obtiene 0.202 en el examen básico y 0.243 en el partido completo.

El modelo de la entrega supera al modelo anterior en los siete partidos. Casi todo el error restante se concentra en posesiones de menos de 3 s, como rechazos y pases de primera que el tracking no permite distinguir con precisión.

En el Barça–Athletic no contamos con etiquetas de referencia para evaluar la posesión. La predicción coincide con el equipo que indica Smart Tagging en el 90.9 % de los frames de sus fases de ataque. La ventana pública de cinco minutos solo la usamos para comprobar que el archivo estuviera bien armado y para entender la definición de posesión, como indica `CHALLENGE.md`.

## Requisitos

- Python 3.9. Probamos la solución con Python 3.9.6 en macOS sobre Apple Silicon.
- Los datos del partido en `data/tactical/` y `data/broadcast/`, con la estructura del paquete original descrita en `data/README.md`. Del broadcast solo usamos el tracking.
- Conexión a internet en la primera ejecución para descargar IDSSE desde Hugging Face (`pysport/idsse-data`).

## Instalación

```bash
python3 -m venv .venv       && .venv/bin/pip install -r requirements-solution.txt
python3 -m venv .venv-torch && .venv-torch/bin/pip install -r requirements-torch.txt
```

Creamos dos entornos porque PyTorch requiere una versión distinta de NumPy.

## Regenerar la entrega

```bash
.venv/bin/python solution/run_all.py --out evaluation/results/Veyon.csv
```

El comando procesa el video, prepara IDSSE, ejecuta las 12 redes guardadas en `solution/models/` y valida el CSV con `evaluation/validate_submission.py`.

- **Salida esperada.** El CSV debe coincidir con el archivo entregado. Su MD5 es `4bb7df9f77f1f7d46a87a68adedc688d`.
- **Tiempo de ejecución.** Aproximadamente 16 minutos desde cero en un MacBook con Apple Silicon.
- **`--skip-perception`** reutiliza los resultados del procesamiento del video.
- **`--retrain`** vuelve a entrenar las redes con GPU mediante MPS o CUDA. Como el entrenamiento en GPU no es determinista, el resultado puede variar ligeramente. Por eso guardamos los pesos.

## Archivos

| Archivo | Función |
|---|---|
| `solution/run_all.py` | Ejecuta el proceso completo. |
| `solution/ball_candidates.py`, `ball_track.py`, `broadcast_sync.py` | Detectan y siguen la pelota. |
| `solution/team_check.py`, `explore/entity_colors.py` | Verifican el equipo mediante el color de la camiseta. |
| `solution/get_idsse.py`, `idsse.py`, `bench.py`, `build_bench.py` | Descargan IDSSE y ajustan sus datos para reproducir las condiciones del tracking del reto. |
| `solution/features.py` | Calcula las variables por frame. |
| `solution/paradas_*.py` | Implementan la simulación de paradas, la red con varias definiciones de posesión, la predicción y la decodificación. |
| `solution/models/` | Contiene los pesos de las 12 redes y las estadísticas de huecos en el seguimiento de la pelota. |

## Datos externos

Bassek, M., Rein, R., Weber, H. y Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer*. Scientific Data, 12, 195. Los datos de IDSSE proceden de la DFL y se distribuyen bajo licencia CC BY 4.0. [Artículo de IDSSE](https://doi.org/10.1038/s41597-025-04505-y)

Los datos del Barça–Athletic se obtienen del paquete del reto y no se incluyen en este repositorio.
