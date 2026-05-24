# CAPÍTULO 2: INGENIERÍA DE DATOS Y AJUSTE CORPORATIVO

## 2.1. El Reto de la Integridad de los Datos Financieros Históricos
La principal barrera para entrenar redes neuronales profundas en series temporales financieras es la calidad intrínseca de los datos. En este proyecto se utilizó la base de datos de BME (Bolsas y Mercados Españoles) para el IBEX 35, con granularidad de alta resolución desde 2018 hasta 2025. El dataset crudo contiene millones de eventos bid/ask y operaciones ejecutadas. Sin embargo, procesar esta información "tal cual" produce distorsiones estructurales masivas que arruinan la capacidad generalizadora de cualquier Inteligencia Artificial.

El problema radica en las acciones corporativas. Cuando una empresa como Iberdrola (IBE) o ACS ejecuta un dividendo flexible (*Scrip Dividend*) o una ampliación de capital liberada, la cotización de la acción se ajusta matemáticamente a la baja en la apertura de la sesión ex-dividendo. Para un operador humano o un inversor a largo plazo, este evento es neutral (ya que recibe nuevas acciones o dinero en efectivo que compensa la caída del precio de la acción). Sin embargo, una Red Neuronal sin contexto interpreta esta discontinuidad como un "Crash" bajista masivo. El modelo aprende falsamente que ciertas formaciones técnicas preceden a colapsos catastróficos del 5%, cuando en realidad se trata de un simple asiento contable.

## 2.2. Metodología Matemática de Ajuste (Adjusted Close)
Para garantizar la integridad del entrenamiento, este trabajo implementa un pipeline de homogeneización histórica, transformando la serie bruta en una serie ajustada mediante el cálculo de un Factor de Ajuste Acumulado ($F_{adj}$).

La fórmula aplicada de forma retroactiva sobre toda la historia previa de una acción tras una acción corporativa $A_t$ en el tiempo $t$ se define como:
$$ F_{adj}(t) = \frac{P_{c}(t-1) - D_{t}}{P_{c}(t-1)} $$
Donde $P_c(t-1)$ es el precio de cierre oficial del día previo al evento, y $D_t$ es el valor del dividendo bruto o el equivalente de dilución. Al multiplicar toda la serie temporal anterior al evento por este factor, se eliminan los saltos ("gaps") artificiales, permitiendo a la red CNN+LSTM capturar los retornos porcentuales puros generados por la oferta y la demanda genuina.

## 2.3. Mapeo Dinámico para evitar el Survivorship Bias
Otro pilar de la fase de datos es evitar el Sesgo de Supervivencia (*Survivorship Bias*). Las composiciones de los índices no son estáticas; cambian trimestralmente. Si entrenamos una red usando exclusivamente los 35 componentes del índice en 2025 y los extrapolamos hacia 2018, la IA estaría aprendiendo de un universo filtrado de "ganadores históricos", empresas que tuvieron el éxito suficiente para sobrevivir 8 años. Esto infla drásticamente los resultados en un backtesting.

Para mitigar este sesgo fatal, se ha programado un algoritmo que lee la composición real (punto en el tiempo) del índice trimestralmente. La IA, tanto en entrenamiento como en simulación operativa, solo tiene permitida la entrada de capital en aquellos *tickers* que formaban parte del índice **exactamente** en la fecha de la inferencia, descartando cualquier ventaja de conocimiento a futuro.

## 2.4. Generación de Etiquetas (*Super Labels*) para Aprendizaje Multiclase
Para transformar el problema de *trading* en un paradigma de aprendizaje supervisado, se estableció una heurística algorítmica para etiquetar ventanas históricas. A partir de una ventana de observación ($H_{obs} = 60$ barras), se analiza un horizonte futuro ($H_{fut} = 48$ barras). El comportamiento del precio en el futuro se etiqueta con 6 clases categóricas (*Super Labels*):

- **0: NONE** (Sin movimiento claro o ruido).
- **1: ALC_CONT** (Continuación Alcista Fuerte).
- **2: BAJ_CONT** (Continuación Bajista Fuerte).
- **3: REV_ALCISTA** (Reversión o rebote al alza).
- **4: REV_BAJISTA** (Reversión o rebote a la baja).
- **5: LATERAL** (Consolidación sin dirección).

El etiquetado no solo evalúa el punto de inicio y el punto final del horizonte de 48 barras, sino que analiza la trayectoria (Máximos, Mínimos, y Drawdown intermedio) para garantizar que, por ejemplo, la etiqueta `ALC_CONT` se asigne únicamente si la trayectoria superó un umbral de rentabilidad mínimo antes de tocar un umbral de stop loss estructural de invalidación. Esto garantiza que la red asocie características técnicas con trayectorias que son matemáticamente rentables en la vida real.
