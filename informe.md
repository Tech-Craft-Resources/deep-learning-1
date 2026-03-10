# Informe de Resultados — Experimentos de Deep Learning
## Nicolas Rodriguez Forero & Daniel Velasco Gonzalez
### Deep Learning - 2026-I

---

[LINK DEL REPOSITORIO DE GITHUB](https://github.com/Tech-Craft-Resources/deep-learning-1.git)

## 1. Predicción del precio de cierre de la acción AMZN

### Descripción del experimento

El objetivo fue construir redes neuronales densas capaces de predecir el precio de cierre de la acción de Amazon (AMZN) para el día siguiente, usando como entrada las últimas *n* jornadas bursátiles (ventana de *n* = 3, 7 o 15 días). Se trabajó con datos históricos diarios desde enero de 2015 hasta diciembre de 2024 (2 516 registros).

Para evaluar el rendimiento de forma rigurosa se aplicó **walk-forward validation** con cuatro períodos de test independientes: 2019, 2020, 2022 y 2023. En cada período, el modelo se entrenó exclusivamente con datos anteriores a la ventana de test, evitando cualquier filtración de información futura.

Se compararon cuatro modelos:

| Modelo | Arquitectura |
|---|---|
| Persistencia (baseline) | Predice `precio[t+1] = precio[t]` |
| DNN Simple | 1 capa oculta (32 neuronas, ReLU) |
| DNN Media | 2 capas ocultas (64 → 32 neuronas) |
| DNN Profunda | 3 capas ocultas (128 → 64 → 32) + Dropout(0.2) |

### Resultados

#### Período 1 — Test: año 2019

| Modelo | Ventana | MAE (USD) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|
| Persistencia | 3 d | 0.94 | 1.26 | 1.06 |
| DNN Simple | 3 d | 1.33 | 1.72 | 1.49 |
| **DNN Media** | **3 d** | **0.99** | **1.31** | **1.11** |
| DNN Profunda | 3 d | 1.14 | 1.50 | 1.28 |
| Persistencia | 7 d | 0.94 | 1.26 | 1.06 |
| DNN Simple | 7 d | 1.69 | 2.20 | 1.90 |
| **DNN Media** | **7 d** | **1.08** | **1.42** | **1.22** |
| DNN Profunda | 7 d | 1.24 | 1.59 | 1.40 |
| Persistencia | 15 d | 0.94 | 1.26 | 1.06 |
| DNN Simple | 15 d | 1.69 | 2.23 | 1.91 |
| **DNN Media** | **15 d** | **1.25** | **1.61** | **1.41** |
| DNN Profunda | 15 d | 1.42 | 1.85 | 1.60 |

#### Período 2 — Test: año 2020 (pandemia COVID-19)

| Modelo | Ventana | MAE (USD) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|
| **Persistencia** | **3 d** | **2.36** | **3.17** | **1.79** |
| DNN Simple | 3 d | 6.97 | 8.36 | 4.80 |
| DNN Media | 3 d | 2.60 | 3.45 | 1.94 |
| DNN Profunda | 3 d | 3.68 | 4.72 | 2.70 |
| **Persistencia** | **7 d** | **2.36** | **3.17** | **1.79** |
| DNN Simple | 7 d | 5.15 | 6.61 | 3.85 |
| DNN Media | 7 d | 2.44 | 3.24 | 1.86 |
| DNN Profunda | 7 d | 4.81 | 6.12 | 3.41 |
| **Persistencia** | **15 d** | **2.36** | **3.17** | **1.79** |
| DNN Simple | 15 d | 4.27 | 5.52 | 3.12 |
| DNN Media | 15 d | 3.32 | 4.37 | 2.51 |
| DNN Profunda | 15 d | 5.11 | 6.41 | 3.78 |

#### Período 3 — Test: año 2022 (corrección del mercado)

| Modelo | Ventana | MAE (USD) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|
| **Persistencia** | **3 d** | **2.94** | **3.98** | **2.37** |
| DNN Simple | 3 d | 3.33 | 4.33 | 2.68 |
| DNN Media | 3 d | 3.57 | 4.56 | 2.87 |
| DNN Profunda | 3 d | 3.80 | 4.74 | 3.11 |
| **Persistencia** | **7 d** | **2.94** | **3.98** | **2.37** |
| DNN Simple | 7 d | 5.88 | 7.38 | 4.83 |
| **DNN Media** | **7 d** | **3.16** | **4.16** | **2.54** |
| DNN Profunda | 7 d | 5.40 | 6.74 | 4.50 |
| **Persistencia** | **15 d** | **2.94** | **3.98** | **2.37** |
| DNN Simple | 15 d | 3.87 | 4.98 | 3.11 |
| DNN Media | 15 d | 4.56 | 5.67 | 3.68 |
| DNN Profunda | 15 d | 6.64 | 8.30 | 5.48 |

#### Período 4 — Test: año 2023

| Modelo | Ventana | MAE (USD) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|
| **Persistencia** | **3 d** | **1.84** | **2.46** | **1.55** |
| DNN Simple | 3 d | 2.14 | 2.80 | 1.80 |
| DNN Media | 3 d | 1.90 | 2.49 | 1.60 |
| DNN Profunda | 3 d | 2.19 | 2.81 | 1.81 |
| **Persistencia** | **7 d** | **1.84** | **2.46** | **1.55** |
| DNN Simple | 7 d | 2.34 | 2.99 | 1.97 |
| DNN Media | 7 d | 1.92 | 2.50 | 1.62 |
| DNN Profunda | 7 d | 2.06 | 2.68 | 1.72 |
| **Persistencia** | **15 d** | **1.84** | **2.46** | **1.55** |
| DNN Simple | 15 d | 6.45 | 7.75 | 5.20 |
| DNN Media | 15 d | 2.34 | 3.03 | 1.97 |
| DNN Profunda | 15 d | 2.62 | 3.28 | 2.15 |

### Interpretación

El resultado más destacado es la **resiliencia del modelo de persistencia**: en la mayoría de los períodos y ventanas, predecir simplemente "mañana el precio será igual al de hoy" resulta tan bueno o mejor que cualquier red neuronal densa. Esto no es un fracaso del deep learning, sino una manifestación de la **hipótesis del mercado eficiente**: los precios de las acciones incorporan rápidamente toda la información disponible, de modo que el histórico reciente tiene escaso poder predictivo adicional respecto al último precio conocido.

El año 2019 (mercado relativamente estable y alcista) fue el único período donde la **DNN Media** logró superar marginalmente al baseline con ventana de 3 días (MAPE 1.11 % frente a 1.06 %). En cambio, el año 2020 ilustra el impacto de eventos imprevisibles como la pandemia de COVID-19: los precios oscilaron de forma extrema, y todos los modelos de red neuronal sufrieron errores muy superiores al baseline. La DNN Simple llegó a un error de 6.97 USD de media diaria (MAPE 4.80 %), más de 2.5 veces peor que la persistencia.

En términos prácticos, un MAPE del 1–2 % sobre AMZN (que cotiza alrededor de los 80–200 USD en el período estudiado) equivale a un error absoluto de entre 0.8 y 4 USD por acción al día. Aunque puede parecer pequeño, no es suficiente para generar señales de trading rentables una vez descontadas comisiones y deslizamiento, lo que confirma la dificultad intrínseca de la predicción bursátil a corto plazo con esta metodología.

---

## 2. Detección de spam con redes neuronales densas

### Descripción del experimento

El objetivo fue clasificar mensajes de texto como *spam* o *ham* (no spam), comparando tres enfoques:

- **DNN:** Red neuronal densa con capas de 128 → 64 neuronas y regularización por Dropout.
- **Random Forest:** Conjunto de árboles de decisión.
- **Naive Bayes Multinomial:** Clasificador probabilístico clásico para texto.

El dataset contenía **656 mensajes** (577 ham, 79 spam), con una proporción de spam del 12 %, lo que representa un escenario de **desbalanceo moderado**. La vectorización del texto se realizó con **TF-IDF** (5 000 características, n-gramas 1–2), ajustado exclusivamente sobre el conjunto de entrenamiento para evitar filtración de datos al test (80 % / 20 %).

### Resultados

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Naive Bayes** | 0.9394 | 0.9000 | 0.5625 | **0.6923** |
| **DNN** | 0.9394 | 1.0000 | 0.5000 | 0.6667 |
| Random Forest | 0.9167 | 1.0000 | 0.3125 | 0.4762 |

### Interpretación

Los tres modelos alcanzan una **accuracy global alta** (entre 91 % y 94 %), pero este número puede resultar engañoso cuando el dataset está desbalanceado: si un clasificador simplemente predijera "todo es ham", obtendría ya un 87 % de accuracy sin detectar ni un solo spam.

La métrica relevante aquí es el **F1-score**, que balancea la precisión (qué fracción de los mensajes marcados como spam son realmente spam) y el recall (qué fracción del spam real fue detectada).

- **Naive Bayes** obtuvo el mejor F1 (0.69), detectando el 56 % del spam con una tasa de falsos positivos baja (10 % de los mensajes marcados como spam eran en realidad legítimos). En un filtro de correo real, esto significa que prácticamente no se bloquearían mensajes legítimos, aunque algo menos de la mitad del spam pasaría desapercibido.

- **La DNN** logró una precisión perfecta (1.00): nunca marcó un correo legítimo como spam. Sin embargo, su recall fue del 50 %, dejando pasar la mitad del spam. Para aplicaciones donde el coste de un falso positivo es alto (bloquear correos importantes), este comportamiento puede ser preferible.

- **Random Forest** tuvo el peor desempeño en recall (31 %), dejando pasar casi el 70 % del spam, lo que lo hace prácticamente inutilizable como filtro en este escenario con los hiperparámetros utilizados.

El rendimiento general por debajo de lo esperado se debe principalmente al **tamaño reducido del dataset**: con solo 79 ejemplos de spam en entrenamiento, los modelos tienen muy pocas muestras para aprender patrones robustos. En aplicaciones reales, los filtros de spam se entrenan con millones de ejemplos y logran F1 superiores a 0.97.

---

## 3. Clasificación de lengua de señas (27 clases)

### Descripción del experimento

El objetivo fue clasificar imágenes de mano en **27 categorías** del alfabeto ASL (*American Sign Language*) mediante redes neuronales densas. El dataset, proveniente de Kaggle (ardamavi/27-class-sign-language-dataset), contiene imágenes en escala de grises de 128×128 píxeles distribuidas de forma aproximadamente balanceada entre las 27 clases.

Dado que una imagen de 128×128×3 genera 49 152 características, conectar directamente una capa densa produciría un modelo con ~25 millones de parámetros y sobreajuste severo con el volumen de datos disponible (~18 000 muestras de entrenamiento). Para mitigar esto se aplicó **PCA** exclusivamente sobre el conjunto de entrenamiento, reduciendo la dimensionalidad de 49 152 a **256 componentes** que explican el 90.4 % de la varianza.

Se entrenó el modelo con mejores resultados:

| Arquitectura | Capas | Regularización |
|---|---|---|
| DNN Regularized | Input(256) → Dense(256, ReLU) → BN → Dropout(0.4) → Dense(128, ReLU) → BN → Dropout(0.3) → Dense(27, Softmax) | BatchNorm + Dropout |

El entrenamiento usó `EarlyStopping` con paciencia de 20 épocas y `ReduceLROnPlateau` con factor 0.5, lo que detuvo el proceso antes de completar las 150 épocas máximas para evitar sobreajuste.

### Resultados

| Modelo | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| DNN Regularized | 0.5578 | 0.5562 | 0.5584 |

**F1 por clase (en test, 173 muestras por clase):**

| Categoría | F1 |
|---|---|
| Clases con mejor desempeño (22, 23, 24) | 0.784 – 0.796 |
| Clases con peor desempeño (4, 6, 2) | 0.371 – 0.406 |

### Interpretación

Un accuracy del **55.8 %** sobre 27 clases equivale a predecir correctamente algo más de la mitad de las imágenes de test. Para poner esto en perspectiva, una clasificación aleatoria obtendría un 3.7 % (1/27), por lo que el modelo aprende un patrón real y significativo, aunque su fiabilidad no es todavía suficiente para un uso aplicado.

La causa principal de los errores es la **similitud visual entre señas**: varios gestos del ASL se diferencian únicamente por la posición del pulgar o la curvatura de los dedos anular y meñique, características sutiles que un vector de características basado en PCA puede no capturar fielmente. La reducción de dimensionalidad, necesaria para evitar sobreajuste, también descarta parte de la información discriminativa fina.

Las clases con mejor F1 (~0.79) probablemente corresponden a gestos con una forma de mano muy característica y fácilmente separable del resto. Las clases con peor F1 (~0.37–0.41) son aquellas cuya representación PCA se solapa con otras clases en el espacio de 256 componentes.

Para superar esta barrera, la arquitectura natural sería una **red convolucional (CNN)**, que aprende filtros espaciales directamente de los píxeles y no requiere proyección previa, permitiendo capturar los detalles locales de los gestos.

---

## 4. Detección de dedos extendidos (clasificación multi-label)

### Descripción del experimento

Usando el mismo dataset de lengua de señas del experimento anterior, el objetivo aquí es diferente: en lugar de identificar *qué seña* se realiza, se busca determinar **qué dedos están extendidos** en cada imagen. Se trata de un problema de **clasificación multi-label** con cinco salidas binarias independientes:

`[Pulgar, Índice, Medio, Anular, Meñique]` → 1 = extendido, 0 = flexionado.

El etiquetado se construyó mediante un **mapeo anatómico** de cada clase ASL a la configuración de dedos correspondiente según el estándar del alfabeto americano.

La arquitectura empleada fue:

```
Input(256) → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
           → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
           → Dense(64, ReLU)
           → Dense(5, Sigmoid)   ← una salida por dedo
```

Se utilizó **binary crossentropy** como función de pérdida (apropiada para multi-label) y se optimizó con Adam con reducción dinámica de la tasa de aprendizaje.

### Resultados globales

| Métrica | Valor |
|---|---|
| Hamming Loss | 0.2645 |
| F1 micro | 0.7966 |
| F1 macro | 0.7834 |
| F1 by-sample | 0.6393 |

### Resultados por dedo

| Dedo | Accuracy | F1 |
|---|---|---|
| **Índice** | 0.8005 | **0.8780** |
| Medio | 0.7395 | 0.8144 |
| Anular | 0.7330 | 0.7901 |
| Meñique | 0.7226 | 0.7827 |
| Pulgar | 0.6819 | 0.6518 |

### Interpretación

El **Hamming Loss de 0.26** significa que, en promedio, el modelo se equivoca en aproximadamente **1.3 de cada 5 dedos** por imagen. Aunque este número puede sonar alto, es importante contextualizarlo: la tarea es intrínsecamente difícil porque la posición de cada dedo es altamente dependiente de la seña que se realiza, y varias señas del ASL comparten configuraciones parciales de dedos muy similares.

El **F1 micro de 0.80** indica que, considerando todas las predicciones binarias del modelo (5 dedos × N imágenes) de forma conjunta, el 80 % de ellas son correctas en términos de precisión-recall. El **F1 by-sample de 0.64** es más exigente: mide si el modelo predice *exactamente* la combinación correcta de dedos para cada imagen, y el 64 % de exactitud en este sentido es un resultado sólido considerando que hay 27 posibles combinaciones anatómicas distintas.

La **disparidad entre dedos** es reveladora: el **índice es el más fácil de detectar** (F1 = 0.88), probablemente porque aparece extendido en muchas señas y su posición es visualmente prominente. El **pulgar es el más difícil** (F1 = 0.65): su posición varía sutilmente entre señas, puede quedar parcialmente oculto por otros dedos, y en el espacio PCA estas diferencias son difíciles de separar.

En conjunto, este experimento demuestra que es posible extraer información anatómica significativa de las imágenes de señas incluso con arquitecturas densas y representaciones PCA, aunque una arquitectura convolucional permitiría capturar mejor la geometría espacial de cada gesto.

---

## Conclusiones generales

| Experimento | Tarea | Mejor resultado |
|---|---|---|
| 1 — AMZN | Regresión (precio del día siguiente) | Baseline de persistencia (MAPE ~1–2 %) |
| 2 — Spam | Clasificación binaria | Naive Bayes (F1 = 0.69) |
| 3 — Lengua de señas | Clasificación 27 clases | DNN Regularized (Acc = 55.8 %) |
| 4 — Dedos extendidos | Multi-label (5 etiquetas) | DNN (F1 micro = 0.80) |

A lo largo de los cuatro experimentos se observan patrones consistentes:

1. **El baseline importa.** En predicción bursátil, la persistencia supera a las redes neuronales en la mayoría de los escenarios porque la información histórica reciente aporta escaso valor predictivo incremental. Evaluar siempre contra un baseline sólido es imprescindible.

2. **El tamaño y el balance del dataset son determinantes.** La detección de spam con 79 ejemplos positivos limita severamente el recall, independientemente del modelo empleado. En clasificación de señas, los ~18 000 ejemplos resultan insuficientes para redes densas sin reducción de dimensionalidad previa.

3. **La regularización mejora la generalización.** En el experimento de lengua de señas, la DNN con BatchNormalization y Dropout fue la que mejor generalizó. En predicción bursátil, sin embargo, la DNN Profunda con Dropout no superó a arquitecturas más simples, sugiriendo que el problema no es de sobreajuste sino de falta de señal predictiva en los datos.

4. **La elección de la arquitectura debe adaptarse a la naturaleza de los datos.** Los problemas de imagen se benefician de capas convolucionales; los problemas de series de tiempo, de arquitecturas recurrentes o de atención. Las redes densas sobre vectores aplanados o PCA son un punto de partida razonable, pero tienen un techo de rendimiento claro.
