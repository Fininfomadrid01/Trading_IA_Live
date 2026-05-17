# TRABAJO DE FIN DE MÃSTER: Deep Learning y GestiÃ³n Monetaria Institucional en el IBEX 35
## Autor: [Tu Nombre] | Tutor: [Nombre del Tutor]
## Fecha: Mayo 2026
## Programa: MÃ¡ster en Finanzas Cuantitativas y Algorithmic Trading

---

# ÃNDICE GENERAL

1. **CAPÃTULO 1: INTRODUCCIÃN Y MARCO TEÃRICO**
   1.1. IntroducciÃ³n
   1.2. JustificaciÃ³n del Uso de Deep Learning
   1.3. Limitaciones Previas y Sesgo de Supervivencia
   1.4. Objetivos del TFM
2. **CAPÃTULO 2: INGENIERÃA DE DATOS Y AJUSTE CORPORATIVO**
   2.1. El Reto de la Integridad de los Datos
   2.2. MetodologÃ­a MatemÃ¡tica de Ajuste (Adjusted Close)
   2.3. Mapeo DinÃ¡mico de Componentes del IBEX 35
   2.4. GeneraciÃ³n de Etiquetas (Super Labels)
3. **CAPÃTULO 3: ARQUITECTURA DE INFERENCIA (DEEP LEARNING)**
   3.1. DiseÃ±o HÃ­brido: CNN + BiLSTM + Attention
   3.2. Proceso de Entrenamiento y RegularizaciÃ³n
   3.3. Inferencia y Umbrales de Confianza (Thresholding)
4. **CAPÃTULO 4: MOTOR DE EJECUCIÃN Y GESTIÃN MONETARIA**
   4.1. El Problema del Dimensionamiento de Capital
   4.2. Filtrado de Riesgo Estructural y Esperanza MatemÃ¡tica
   4.3. Agrupamiento por Tiers de Liquidez
## 4.4. Generalización Internacional: Presupuestación de Riesgo (Risk Budgeting)
Aunque en los mercados internacionales (NASDAQ 100, FTSE 100, Criptomonedas y Futuros sobre Commodities) no exista un riesgo real de *slippage* debido a su colosal volumen y profundidad diaria —haciendo innecesaria la restricción de liquidez por Tiers—, sí resulta obligatorio limitar el capital asignado en base a su perfil de **Volatilidad y Correlación**. En finanzas cuantitativas profesionales, esta estrategia de control se denomina **Presupuestación de Riesgo (Risk Budgeting)**.

Si el sistema tratase a todos los activos por igual asignándoles un presupuesto máximo (Tier 1 del 15%), la cartera global sufriría desplomes (*drawdowns*) extremos e inaceptables cuando los activos de alta volatilidad (como el Bitcoin o el Petróleo) experimentasen correcciones bruscas. Por consiguiente, el motor global macro de la plataforma asigna de forma nativa equivalencias de Tiers ajustadas a la volatilidad histórica de cada tipo de activo:
*   **Equivalencia a Tier 3 (Exposición Base del 4% - 5%):** Se aplica a **Criptoactivos (ETFs IBIT / ETHA)**. No es por falta de liquidez (poseen profundidad de sobra), sino para neutralizar su volatilidad extrema. Limitar la posición al 4% actúa como un amortiguador, garantizando que un latigazo del -15% en el Bitcoin no altere de forma catastrófica la curva de capital de la cuenta.
*   **Equivalencia a Tier 2 (Exposición Base del 8%):** Se aplica a **Materias Primas (Oro/Petróleo)** y a la **Bolsa de Londres (FTSE 100)**. Ofrecen una volatilidad media-baja y actúan como excelentes diversificadores globales o de activos reales.
*   **Equivalencia a Tier 1 (Exposición Base del 10% - 15%):** Se aplica a las **Blue Chips del NASDAQ 100 e IBEX 35**. Constituyen el motor de crecimiento y el pilar direccional de renta variable principal de la cartera.

Esta sofisticación dota al sistema de una robustez matemática de grado institucional, permitiendo su despliegue y preservación de capital multiactivo.

## 4.5. Teoría Anti-Martingala Calibrada
   4.5. Teoría Anti-Martingala Calibrada
   4.6. Resolución del "Cash Drag" e Integridad Contable
5. **CAPÃTULO 5: RESULTADOS EMPÃRICOS (BACKTESTING)**
   5.1. DiseÃ±o Experimental
   5.2. Plano de la SeÃ±al: Esperanza MatemÃ¡tica
   5.3. Plano de la Cartera: Rendimiento y Perfil de Riesgo
   5.4. AnÃ¡lisis de FricciÃ³n y Sharpe Ratio
   5.5. GeneralizaciÃ³n Internacional Cruzada (EE.UU. y Reino Unido)
   5.6. La Cartera HÃ­brida Global Macro (AuditorÃ­a PrÃ¡ctica 2025-2026)
6. **CAPÃTULO 6: IMPLEMENTACIÃN EN PRODUCCIÃN Y AUTOMATIZACIÃN CLOUD**
   6.1. Arquitectura del Pipeline Live
   6.2. SincronizaciÃ³n Nube-Local (GitHub Actions)
   6.3. IntegraciÃ³n con Interactive Brokers (IBKR API)
   6.4. El Dashboard de Toma de Decisiones
7. **CAPÃTULO 7: CONCLUSIONES Y FUTURAS LÃNEAS DE INVESTIGACIÃN**
   7.1. SÃ­ntesis y EvaluaciÃ³n de Resultados
   7.2. Futuras LÃ­neas de InvestigaciÃ³n

---


## 1.1. IntroducciÃ³n
El constante avance de la capacidad computacional y la disponibilidad masiva de datos estructurados han transformado radicalmente los mercados financieros. Durante dÃ©cadas, el anÃ¡lisis de series temporales financieras estuvo dominado por modelos estadÃ­sticos lineales, como ARIMA o GARCH, los cuales asumen que las series de precios presentan estacionariedad y distribuciones normales. Sin embargo, la literatura empÃ­rica moderna ha demostrado sistemÃ¡ticamente que los mercados bursÃ¡tiles son sistemas dinÃ¡micos complejos, caracterizados por alta volatilidad, *fat tails* (colas pesadas), heterocedasticidad y fuertes componentes de ruido no lineal.

En este contexto, el presente Trabajo de Fin de MÃ¡ster (TFM) propone el desarrollo, entrenamiento y validaciÃ³n de una arquitectura hÃ­brida de *Deep Learning* (Aprendizaje Profundo) capaz de extraer de forma autÃ³noma patrones complejos de la microestructura del mercado bursÃ¡til espaÃ±ol (IBEX 35). Para ello, se plantea un enfoque superestructurado que aÃºna Redes Neuronales Convolucionales (CNN) unidimensionales, orientadas a la extracciÃ³n de caracterÃ­sticas topolÃ³gicas locales de los precios, con Redes de Memoria a Corto y Largo Plazo Bidireccionales (Bi-LSTM), diseÃ±adas para capturar las dependencias temporales y la secuencia lÃ³gica del mercado. AdemÃ¡s, se integra un Mecanismo de AtenciÃ³n (*Attention Mechanism*) que permite al modelo ponderar dinÃ¡micamente quÃ© momentos del pasado son mÃ¡s relevantes para la predicciÃ³n futura.

## 1.2. JustificaciÃ³n del Uso de Deep Learning en Mercados Financieros
A diferencia de los modelos tradicionales de *Machine Learning* como Random Forest o Support Vector Machines (SVM), que requieren de un laborioso proceso de ingenierÃ­a de caracterÃ­sticas (*feature engineering*) guiado por el ser humano, el Deep Learning permite el aprendizaje de representaciones de manera jerÃ¡rquica y directa desde los datos crudos (*raw data*).

Esta caracterÃ­stica es fundamental en la operativa de alta y media frecuencia, donde los patrones subyacentes son efÃ­meros e invisibles al ojo humano. En lugar de predefinir indicadores tÃ©cnicos clÃ¡sicos (RSI, MACD), el sistema propuesto ingiere tensores multidimensionales compuestos por datos OHLCV (Open, High, Low, Close, Volume) sin procesar, delegando en las capas convolucionales la responsabilidad de descubrir los predictores alfa mÃ¡s eficientes.

## 1.3. Limitaciones Previas y el Problema de la Supervivencia
Gran parte de la investigaciÃ³n acadÃ©mica previa en predicciÃ³n financiera sufre de graves sesgos metodolÃ³gicos que invalidan sus resultados fuera del entorno de laboratorio (*out-of-sample*). El mÃ¡s crÃ­tico de ellos es el **Sesgo de Supervivencia** (*Survivorship Bias*) y la **ContaminaciÃ³n por Eventos Corporativos**.

Es prÃ¡ctica comÃºn utilizar las bases de datos histÃ³ricas de Ã­ndices bursÃ¡tiles asumiendo que los componentes actuales han sido los mismos histÃ³ricamente. Esto ignora a las empresas que quebraron o fueron excluidas del Ã­ndice, sesgando artificialmente el rendimiento hacia los activos "ganadores". AdemÃ¡s, los eventos de diluciÃ³n de capital, como el reparto de dividendos (especialmente el formato *Scrip Dividend* tan comÃºn en EspaÃ±a) o los desdoblamientos de acciones (*Splits*), generan caÃ­das artificiales de precios que engaÃ±an a las redes neuronales, induciÃ©ndolas a identificar patrones bajistas inexistentes.

## 1.4. Objetivos del Trabajo de Fin de MÃ¡ster
Para abordar estos desafÃ­os metodolÃ³gicos, este TFM establece los siguientes objetivos:
1. **AuditorÃ­a y CorrecciÃ³n de Datos:** Desarrollar un pipeline metodolÃ³gico robusto que ajuste matemÃ¡ticamente la serie histÃ³rica intradiaria del IBEX 35 (2018-2025) aislando el efecto del sesgo de supervivencia y de las acciones corporativas.
2. **Desarrollo de Arquitectura Neuronal:** DiseÃ±ar e implementar un modelo de clasificaciÃ³n multiclase (CNN + BiLSTM + Attention) optimizado en GPU que prediga las continuaciones e inversiones de tendencia con un umbral de confianza estadÃ­sticamente significativo.
3. **ValidaciÃ³n Pluridimensional:** Evaluar el rendimiento del modelo no solo a travÃ©s de mÃ©tricas clÃ¡sicas de Machine Learning (F1-Score, Precision, Recall), sino traduciendo estas probabilidades a **Esperanza MatemÃ¡tica** operativa (Plano de la SeÃ±al).
4. **GestiÃ³n Monetaria Institucional:** Acoplar el modelo predictivo a un motor de ejecuciÃ³n (Plano de la Cartera) que simule de forma hiperrealista la asunciÃ³n de riesgos, incorporando restricciones de liquidez por *Tiers*, cobro de comisiones dinÃ¡micas y la aplicaciÃ³n de reinversiones por la TeorÃ­a Anti-Martingala calibrada.


---

# CAPÃTULO 2: INGENIERÃA DE DATOS Y AJUSTE CORPORATIVO
> **Script asociado:** `Codigo_06_Actualizador_Yahoo.py`

## 2.1. El Reto de la Integridad de los Datos Financieros HistÃ³ricos

La principal barrera para entrenar redes neuronales profundas en series temporales financieras es la calidad intrÃ­nseca de los datos. En este proyecto se utilizÃ³ la base de datos de BME (Bolsas y Mercados EspaÃ±oles) para el IBEX 35, con granularidad de alta resoluciÃ³n desde 2018 hasta 2025. El dataset crudo contiene millones de eventos bid/ask y operaciones ejecutadas. Sin embargo, procesar esta informaciÃ³n "tal cual" produce distorsiones estructurales masivas que arruinan la capacidad generalizadora de cualquier Inteligencia Artificial.

El problema radica en las acciones corporativas. Cuando una empresa como Iberdrola (IBE) o ACS ejecuta un dividendo flexible (*Scrip Dividend*) o una ampliaciÃ³n de capital liberada, la cotizaciÃ³n de la acciÃ³n se ajusta matemÃ¡ticamente a la baja en la apertura de la sesiÃ³n ex-dividendo. Para un operador humano o un inversor a largo plazo, este evento es neutral (ya que recibe nuevas acciones o dinero en efectivo que compensa la caÃ­da del precio de la acciÃ³n). Sin embargo, una Red Neuronal sin contexto interpreta esta discontinuidad como un "Crash" bajista masivo. El modelo aprende falsamente que ciertas formaciones tÃ©cnicas preceden a colapsos catastrÃ³ficos del 5%, cuando en realidad se trata de un simple asiento contable.

## 2.2. MetodologÃ­a MatemÃ¡tica de Ajuste (Adjusted Close)
Para garantizar la integridad del entrenamiento, este trabajo implementa un pipeline de homogeneizaciÃ³n histÃ³rica, transformando la serie bruta en una serie ajustada mediante el cÃ¡lculo de un Factor de Ajuste Acumulado ($F_{adj}$).

La fÃ³rmula aplicada de forma retroactiva sobre toda la historia previa de una acciÃ³n tras una acciÃ³n corporativa $A_t$ en el tiempo $t$ se define como:
$$ F_{adj}(t) = \frac{P_{c}(t-1) - D_{t}}{P_{c}(t-1)} $$
Donde $P_c(t-1)$ es el precio de cierre oficial del dÃ­a previo al evento, y $D_t$ es el valor del dividendo bruto o el equivalente de diluciÃ³n. Al multiplicar toda la serie temporal anterior al evento por este factor, se eliminan los saltos ("gaps") artificiales, permitiendo a la red CNN+LSTM capturar los retornos porcentuales puros generados por la oferta y la demanda genuina.

## 2.3. Mapeo DinÃ¡mico para evitar el Survivorship Bias
Otro pilar de la fase de datos es evitar el Sesgo de Supervivencia (*Survivorship Bias*). Las composiciones de los Ã­ndices no son estÃ¡ticas; cambian trimestralmente. Si entrenamos una red usando exclusivamente los 35 componentes del Ã­ndice en 2025 y los extrapolamos hacia 2018, la IA estarÃ­a aprendiendo de un universo filtrado de "ganadores histÃ³ricos", empresas que tuvieron el Ã©xito suficiente para sobrevivir 8 aÃ±os. Esto infla drÃ¡sticamente los resultados en un backtesting.

Para mitigar este sesgo fatal, se ha programado un algoritmo que lee la composiciÃ³n real (punto en el tiempo) del Ã­ndice trimestralmente. La IA, tanto en entrenamiento como en simulaciÃ³n operativa, solo tiene permitida la entrada de capital en aquellos *tickers* que formaban parte del Ã­ndice **exactamente** en la fecha de la inferencia, descartando cualquier ventaja de conocimiento a futuro.

## 2.4. GeneraciÃ³n de Etiquetas (*Super Labels*) para Aprendizaje Multiclase
Para transformar el problema de *trading* en un paradigma de aprendizaje supervisado, se estableciÃ³ una heurÃ­stica algorÃ­tmica para etiquetar ventanas histÃ³ricas. A partir de una ventana de observaciÃ³n ($H_{obs} = 60$ barras), se analiza un horizonte futuro ($H_{fut} = 48$ barras). El comportamiento del precio en el futuro se etiqueta con 6 clases categÃ³ricas (*Super Labels*):

### 2.4.1. TaxonomÃ­a de Patrones y ReestructuraciÃ³n en Super-Familias
El proceso de generaciÃ³n de etiquetas es uno de los pilares de innovaciÃ³n de este trabajo. En lugar de predecir retornos binarios, el sistema identifica 10 morfologÃ­as tÃ©cnicas clÃ¡sicas que posteriormente se agrupan en 6 "Super-Familias Operativas" para la toma de decisiones. 

Esta reestructuraciÃ³n, ejecutada en la **Fase 4** del pipeline (`fase4_reestructura.py`), permite al modelo consolidar el conocimiento de patrones con implicaciones similares, mejorando la robustez estadÃ­stica de la inferencia:

| PatrÃ³n Original (10 Clases) | Super-Familia (6 Clases) | Significado Operativo |
| :--- | :--- | :--- |
| **CA** (Canal Alcista) | **ALC_CONT** | ContinuaciÃ³n de Tendencia Alcista |
| **CB** (Canal Bajista) | **BAJ_CONT** | ContinuaciÃ³n de Tendencia Bajista |
| **DS**, **HCHI**, **SC** | **REV_ALCISTA** | ReversiÃ³n Alcista (Suelos y Copas) |
| **DT**, **HCH**, **TC** | **REV_BAJISTA** | ReversiÃ³n Bajista (Techos y CuÃ±as) |
| **REC** (RectÃ¡ngulo) | **LATERAL** | ConsolidaciÃ³n / Rango Lateral |
| **NONE** | **NONE** | Ruido Blanco / Sin PatrÃ³n Claro |

Este enfoque hÃ­brido garantiza que la red neuronal CNN+BiLSTM no solo aprenda la direcciÃ³n del precio, sino la **morfologÃ­a del volumen y la volatilidad** asociada a estructuras de acumulaciÃ³n y distribuciÃ³n institucional.

---

# CAPÃTULO 3: ARQUITECTURA DE INFERENCIA (DEEP LEARNING)
> **Script asociado:** `Codigo_01_Arquitectura_Neuronal.py` e `Inferencia_GPU.py`

## 3.1. DiseÃ±o HÃ­brido: CNN + BiLSTM + Attention

El nÃºcleo analÃ­tico de este trabajo reside en una arquitectura de Red Neuronal Profunda diseÃ±ada especÃ­ficamente para el procesamiento de series temporales no estacionarias. El modelo adopta un diseÃ±o hÃ­brido:

### 3.1.1. Rama Convolucional Multi-escala (Multi-scale CNN 1D)
En lugar de una capa convolucional Ãºnica, el modelo utiliza tres ramas en paralelo con diferentes tamaÃ±os de kernel ($k=3, 7, 15$). Esto permite capturar patrones de diferentes frecuencias:
- **Kernel 3:** Captura el ruido de alta frecuencia y micro-impulsos de precio.
- **Kernel 15:** Identifica formaciones grÃ¡ficas mÃ¡s lentas y tendencias locales.
Cada rama aplica 64 filtros, seguidos de **Batch Normalization** y activaciÃ³n **ReLU**. La salida de estas tres ramas se concatena para formar un mapa de caracterÃ­sticas enriquecido.

### 3.1.2. Memoria Bidireccional (BiLSTM)
Los mapas de caracterÃ­sticas se inyectan en una capa BiLSTM de 64 unidades. Al ser bidireccional, el modelo procesa la secuencia tanto de pasado a presente como de "presente a pasado" dentro de la ventana de observaciÃ³n (60 barras).

**JustificaciÃ³n de la Bidireccionalidad en Finanzas:**
Aunque el trading es intrÃ­nsecamente causal (no podemos conocer el futuro), el uso de BiLSTM en ventanas fijas de historia no introduce *look-ahead bias*. Su funciÃ³n es permitir que la red comprenda la **morfologÃ­a completa de un patrÃ³n** dentro de la ventana. Para una evaluaciÃ³n multidimensional integral, el modelo procesa:
6.  **Volatilidad LogarÃ­tmica (STD):** Captura la "nerviosismo" del mercado.
7.  **Log-Retorno del Volumen:** Crucial para identificar la convicciÃ³n institucional detrÃ¡s de cada movimiento de precio.
8.  **Tendencia Local (SMA):** Contextualiza el patrÃ³n dentro de la estructura inmediata.

La informaciÃ³n de las velas posteriores (que ya han ocurrido en relaciÃ³n al momento de la inferencia) proporciona un contexto vital para identificar estructuras simÃ©tricas, como Dobles Suelos o Hombros-Cabeza-Hombro, que una LSTM unidireccional podrÃ­a tardar mÃ¡s en reconocer.

### 3.1.3. Mecanismo de AtenciÃ³n (Dot-Product Attention)
Para evitar que el modelo se pierda en periodos de ruido, se implementÃ³ una capa de atenciÃ³n que pondera la relevancia de cada paso temporal $t$:
1. Se calcula un *score* de energÃ­a $e_t$ para cada estado oculto $h_t$.
2. Se normaliza mediante una funciÃ³n Softmax para obtener los pesos de atenciÃ³n $\alpha_t$.
3. Se genera un vector de contexto $c$ que es la suma ponderada de todos los estados.
$$ c = \sum_{t=1}^{T} \alpha_t h_t $$

## 3.2. HiperparÃ¡metros de Entrenamiento
El entrenamiento se realizÃ³ en un entorno GPU NVIDIA, utilizando los siguientes parÃ¡metros de control:

| ParÃ¡metro | Valor | FunciÃ³n |
| :--- | :--- | :--- |
| Optimizer | Adam | OptimizaciÃ³n estocÃ¡stica con momentum |
| Learning Rate | 0.001 | Velocidad de convergencia inicial |
| Batch Size | 64 | TamaÃ±o del paquete de entrenamiento |
| Dropout | 0.40 | PrevenciÃ³n de Overfitting (regularizaciÃ³n) |
| Early Stopping | 15 epochs | DetenciÃ³n por estancamiento en validaciÃ³n |

---


## 3.3. Inferencia y Umbrales de Confianza (Thresholding)
Tras una fase inicial de entrenamiento sobre 10 morfologÃ­as granulares, el modelo final implementado en este trabajo fue **re-entrenado** (Fase 4) para optimizar la clasificaciÃ³n hacia las **6 Super-Familias Operativas**. Por tanto, la capa de salida final consiste en una neurona Densa con activaciÃ³n *Softmax* de 6 dimensiones.

El sistema exige un rigor estadÃ­stico severo para la toma de decisiones: se configurÃ³ un Umbral de Confianza (`PROB_UMBRAL = 0.93`). La red debe estar segura al menos en un 93% de que el patrÃ³n inminente pertenece a la clase `ALC_CONT` (ContinuaciÃ³n Alcista Fuerte) para que el evento se considere matemÃ¡ticamente digno de inversiÃ³n. Este umbral, calibrado especÃ­ficamente para cada mercado (desde el 92.3% en IBEX hasta el 95.4% en NASDAQ), elimina mÃ¡s del 80% de las inferencias ruidosas, transformando un sistema especulativo en un detector de francotirador estadÃ­stico.


---

# CAPÃTULO 4: MOTOR DE EJECUCIÃN Y GESTIÃN MONETARIA AVANZADA
> **Script asociado:** `Codigo_03_Backtesting_Capital_Real.py`

## 4.1. El Problema ClÃ¡sico del Dimensionamiento de Capital

Uno de los fallos metodolÃ³gicos mÃ¡s recurrentes en la literatura acadÃ©mica de predicciÃ³n bursÃ¡til es asumir liquidez infinita y capital inagotable. Es comÃºn que los modelos teÃ³ricos simulen inversiones idÃ©nticas (e.g., apostar 10.000â¬ a cada seÃ±al producida) sin contemplar si en ese momento temporal el sistema posee efectivo disponible, o si el activo subyacente soporta ese nivel de absorciÃ³n de liquidez sin generar *slippage* severo.

Para alinear el presente modelo predictivo con la rigurosidad institucional y acadÃ©mica requerida, se ha desarrollado un Motor de GestiÃ³n Monetaria de "Fase 2" (*Phase 2: Money Management Engine*). Este motor impone una estricta separaciÃ³n metodolÃ³gica entre el **Plano de la SeÃ±al** (cuÃ¡ntas veces acierta la IA y con quÃ© rentabilidad pura) y el **Plano de la Cartera** (cÃ³mo se despliega de forma inteligente y jerÃ¡rquica el capital finito de la cuenta para protegerse de la quiebra).

## 4.2. Filtrado de RestricciÃ³n Estructural y Esperanza MatemÃ¡tica
Antes de arriesgar capital, toda seÃ±al de la inteligencia artificial debe ser evaluada probabilÃ­sticamente. La Esperanza MatemÃ¡tica ($EV$) mide el retorno real esperado por operaciÃ³n integrando tanto la precisiÃ³n heurÃ­stica como el Ratio Riesgo-Beneficio:

$$ EV = (WR \times AvgWin) + ((1 - WR) \times AvgLoss) $$

Adicionalmente, se programÃ³ una RestricciÃ³n de Riesgo Estructural (*Structural Risk Filter*). Un algoritmo genÃ©tico previo calculÃ³ que el riesgo Ã³ptimo de ruina para este mercado exigÃ­a una volatilidad base de invalidaciÃ³n (`min_struct_risk`) del 4.7%. Por consiguiente, el motor rechaza cualquier seÃ±al que no permita establecer matemÃ¡ticamente un umbral de stop loss al -4.7% de distancia, purificando la cartera de operaciones con un perfil de riesgo asimÃ©trico desfavorable.

## 4.3. GestiÃ³n de Liquidez e Impacto de Mercado (Tier Allocation)
Para evitar el "deslizamiento" (*slippage*) en activos de baja capitalizaciÃ³n, el motor de ejecuciÃ³n integra un anÃ¡lisis de **Volumen Efectivo**. El IBEX 35 es extremadamente asimÃ©trico; invertir el mismo capital en el Banco Santander que en Sacyr expondrÃ­a a la cartera a riesgos de ejecuciÃ³n inaceptables.

Para resolver este problema de manera limpia y sin incurrir en sesgo de conocimiento a futuro (*look-ahead bias*), el motor de capital ejecuta una ponderaciÃ³n dinÃ¡mica trimestral:

1. Al inicio de cada trimestre, el algoritmo analiza el volumen mediano negociado por cada *ticker* **durante el trimestre anterior**.
2. Los valores se ordenan de mayor a menor liquidez histÃ³rica y se fraccionan matemÃ¡ticamente en tres tercios (*Tiers*).
3. La cuenta asigna presupuestos fijos de exposiciÃ³n en base a este ranking:
   - **Tier 1 (Alta Liquidez):** ExposiciÃ³n base del **15%** de la Equidad Total.
   - **Tier 2 (Media Liquidez):** ExposiciÃ³n base del **8%**.
   - **Tier 3 (Baja Liquidez):** ExposiciÃ³n base del **4%**.

### 4.3.1. CÃ¡lculo del Volumen Mediano Trimestral
Para asignar cada activo a un Tier sin incurrir en *look-ahead bias*, el sistema ejecuta el siguiente algoritmo al final de cada trimestre $Q$:
1. Extrae la serie de volumen diario $V_d$ para los 35 componentes.
2. Calcula el volumen efectivo: $VE = P_{cierre} \times V_d$.
3. Calcula la mediana del $VE$ para los Ãºltimos 63 dÃ­as de negociaciÃ³n (un trimestre).
4. Clasifica los activos en Percentiles: [0-33%] Tier 3, [34-66%] Tier 2, [67-100%] Tier 1.

| Tier | Liquidez | Alloc Base | Perfil de Activo |
| :--- | :--- | :--- | :--- |
| 1 | Muy Alta | 15% | Blue Chips (Santander, Iberdrola, Inditex) |
| 2 | Media | 8% | Mid Caps (Acerinox, Bankinter) |
| 3 | Baja | 4% | Small Caps / Chicharros (Sacyr, Solaria) |

## 4.4. Generalización Internacional: Presupuestación de Riesgo (Risk Budgeting)
Aunque en los mercados internacionales (NASDAQ 100, FTSE 100, Criptomonedas y Futuros sobre Commodities) no exista un riesgo real de *slippage* debido a su colosal volumen y profundidad diaria —haciendo innecesaria la restricción de liquidez por Tiers—, sí resulta obligatorio limitar el capital asignado en base a su perfil de **Volatilidad y Correlación**. En finanzas cuantitativas profesionales, esta estrategia de control se denomina **Presupuestación de Riesgo (Risk Budgeting)**.

Si el sistema tratase a todos los activos por igual asignándoles un presupuesto máximo (Tier 1 del 15%), la cartera global sufriría desplomes (*drawdowns*) extremos e inaceptables cuando los activos de alta volatilidad (como el Bitcoin o el Petróleo) experimentasen correcciones bruscas. Por consiguiente, el motor global macro de la plataforma asigna de forma nativa equivalencias de Tiers ajustadas a la volatilidad histórica de cada tipo de activo:
*   **Equivalencia a Tier 3 (Exposición Base del 4% - 5%):** Se aplica a **Criptoactivos (ETFs IBIT / ETHA)**. No es por falta de liquidez (poseen profundidad de sobra), sino para neutralizar su volatilidad extrema. Limitar la posición al 4% actúa como un amortiguador, garantizando que un latigazo del -15% en el Bitcoin no altere de forma catastrófica la curva de capital de la cuenta.
*   **Equivalencia a Tier 2 (Exposición Base del 8%):** Se aplica a **Materias Primas (Oro/Petróleo)** y a la **Bolsa de Londres (FTSE 100)**. Ofrecen una volatilidad media-baja y actúan como excelentes diversificadores globales o de activos reales.
*   **Equivalencia a Tier 1 (Exposición Base del 10% - 15%):** Se aplica a las **Blue Chips del NASDAQ 100 e IBEX 35**. Constituyen el motor de crecimiento y el pilar direccional de renta variable principal de la cartera.

Esta sofisticación dota al sistema de una robustez matemática de grado institucional, permitiendo su despliegue y preservación de capital multiactivo.

## 4.5. Teoría Anti-Martingala Calibrada
El multiplicador de riesgo $M_T$ permite que el sistema "aprenda" de sus rachas ganadoras. La fÃ³rmula de actualizaciÃ³n tras cada operaciÃ³n $n$ es:
$$ M_{T, n+1} = \text{clip}(M_{T, n} \times \text{Factor}, 0.5, 2.0) $$
- **Si el trade es ganador:** Factor = 1.17
- **Si el trade es perdedor:** Factor = 0.96

Esta asimetrÃ­a (17% de subida vs 4% de bajada) estÃ¡ diseÃ±ada para que el sistema necesite pocas rachas ganadoras para alcanzar la exposiciÃ³n mÃ¡xima, pero sea muy lento en reducir el riesgo, protegiendo la curva de capital de "latigazos" de volatilidad.

## 4.6. Resolución del "Cash Drag" e Integridad Contable
Finalmente, el modelo aborda el dilema clÃ¡sico del *Cash Drag* (capital no invertido) bajo la restricciÃ³n del efectivo. En situaciones donde mÃºltiples seÃ±ales coinciden en el mismo dÃ­a temporal, el modelo simula el tiempo exacto. Ordena las oportunidades priorizando **estrictamente por la Probabilidad emitida por la CNN**, calculando el tamaÃ±o ($Size = Equidad \times Tier \times M_T$). Si la "Caja Restante" es matemÃ¡ticamente inferior al tamaÃ±o requerido, el simulador institucional rechaza la operaciÃ³n instantÃ¡neamente, replicando la escasez de margen real y validando su credibilidad empÃ­rica total.

## 4.7. Filtro de Régimen de Mercado (Regime Switching - SMA 200)
Para blindar el sistema contra los mercados bajistas (*Bear Markets*), se implementÃ³ un filtro de rÃ©gimen binario basado en la **Media MÃ³vil Simple de 200 sesiones (SMA 200)** aplicada al Ã­ndice de referencia (e.g., IBEX 35, NASDAQ 100).

Este filtro actÃºa como un interruptor maestro para la estrategia:
*   **Estado BULL (Precio > SMA 200):** El sistema opera con normalidad, ejecutando todas las seÃ±ales de alta confianza emitidas por la red neuronal. Se asume que el sesgo del mercado es favorable y las probabilidades de continuaciÃ³n alcista son mÃ¡ximas.
*   **Estado BEAR (Precio < SMA 200):** El sistema entra en modo de defensa. Se ignoran las seÃ±ales de compra o se restringen a un multiplicador de riesgo mÃ­nimo (0.5x), priorizando la preservaciÃ³n del capital frente a la rentabilidad.

La integraciÃ³n de este filtro de "sentido comÃºn macroeconÃ³mico" permitiÃ³ al modelo evitar los desplomes mÃ¡s severos de 2020 y 2022, concentrando el despliegue de capital Ãºnicamente en ventanas de alta probabilidad de Ã©xito estructural.

---


# CAPÃTULO 5: RESULTADOS EMPÃRICOS (BACKTESTING)
> **Scripts asociados:** `Codigo_04_Comparativa_Completa.py` y `Codigo_05_Analisis_Estadistico.py`

## 5.1. IntroducciÃ³n al DiseÃ±o Experimental

Para validar cientÃ­ficamente la ventaja (*edge*) de la arquitectura de *Deep Learning* propuesta, se ejecutÃ³ un diseÃ±o experimental de backtesting de alta fidelidad operando bajo la modalidad *Event-Driven* (guiado por eventos). Este paradigma, a diferencia de los modelos vectorizados convencionales, previene cualquier forma de conocimiento futuro al simular la entrada de ticks o barras temporalmente de forma estricta. 

La ventana de evaluaciÃ³n comprendiÃ³ desde abril de 2018 hasta inicios de 2025, un horizonte macroeconÃ³mico complejo que encapsulÃ³ el colapso bursÃ¡til generado por la pandemia de la COVID-19 (2020), la posterior recuperaciÃ³n incentivada por los bancos centrales, y la crisis inflacionaria de 2022. Todo el conjunto de datos se ajustÃ³ matemÃ¡ticamente por acciones corporativas, eliminando las rentabilidades ficticias del *Survivorship Bias*. 

AdemÃ¡s, se incorporÃ³ la fricciÃ³n realista del mercado gravando cada operaciÃ³n con la estructura de comisiones por tramos institucionales de *Interactive Brokers* (0.05% del nocional o mÃ­nimo 3.00 EUR por pierna), asegurando que los rendimientos reportados constituyeran mÃ©tricas netas limpias, descontadas de cualquier ilusiÃ³n aritmÃ©tica teÃ³rica.

Los resultados de las 1.926 predicciones validadas y efectivamente disparadas durante el horizonte de *Out-of-Sample* confirmaron un salto cualitativo tras la reestructuraciÃ³n de clases:
- **Accuracy Global (OOS):** 57.36%
- **Recall (Clase ALC_CONT):** 82.00%
- **F1-Score Macro:** 50.79%

### 5.2.1. Desglose de la Esperanza MatemÃ¡tica ($EV$)
El hallazgo cardinal de este plano reside en la **Esperanza MatemÃ¡tica del modelo**. Para un inversor institucional, el valor de una seÃ±al no es su porcentaje de acierto absoluto, sino su rentabilidad esperada por cada euro arriesgado. El sistema demostrÃ³ un *Edge* positivo de **+0.297% de retorno esperado por operaciÃ³n**.

Este valor se deriva de la siguiente distribuciÃ³n de retornos en el plano de la seÃ±al:
- **Tasa de Acierto Operativa ($W$):** ~51.8% (ajustada tras costes).
- **Retorno Medio Ganador ($AvgWin$):** +1.52%
- **Retorno Medio Perdedor ($AvgLoss$):** -1.01%
$$ EV = (0.518 \times 1.52\%) - (0.482 \times 1.01\%) = +0.297\% $$

### 5.2.2. El Valor EstratÃ©gico del Recall (82%)
Resulta imperativo destacar el **Recall del 82%** en las seÃ±ales de continuaciÃ³n alcista. En el trading de sistemas, el Recall mide la capacidad del modelo para "no perderse" un movimiento ganador. 
Un Recall tan elevado indica que la arquitectura CNN+BiLSTM actÃºa como un "radar de tendencias" extremadamente sensible. Aunque esto pueda incrementar ligeramente los falsos positivos (bajando la precisiÃ³n), el **Umbral de Confianza del 93%** actÃºa como un filtro de calidad final. Esta combinaciÃ³n permite al sistema capturar la gran mayorÃ­a de los *rallies* del IBEX 35, asegurando que el capital estÃ© invertido en los momentos de mayor inercia alcista, lo cual es la base de la rentabilidad acumulada del 69% observada en la cartera.

## 5.3. Plano de la Cartera: Rendimiento y Perfil de Riesgo
Una predicciÃ³n eficiente carece de utilidad si su despliegue destruye la cuenta en rachas negativas. Al integrar la probabilidad predictiva con el motor de *Tiers* de Liquidez (para prevenir el colapso de precios bid-ask en *Small Caps*) y la tÃ©cnica *Anti-Martingala Calibrada* [0.5x - 2.0x] (para la reinversiÃ³n acelerada en regÃ­menes de alta predictibilidad), se gestÃ³ un perfil de cartera diametralmente mÃ¡s robusto.

El Capital Inicial se fijÃ³ en 100,000.00 EUR. Tras procesar los 8 aÃ±os de histÃ³rico:
### 5.3.1. Rendimiento de la Propuesta Final: Regime Switching (BULL/BEAR)
Tras un proceso de optimizaciÃ³n iterativo, la investigaciÃ³n concluyÃ³ que la mÃ¡xima eficiencia del capital se alcanza mediante una estrategia de **Regime Switching** (Cambio de RÃ©gimen). Este algoritmo introduce un filtro de salud macro-estructural que altera dinÃ¡micamente el comportamiento del bot segÃºn el estado de los Ã­ndices de referencia (IBEX 35, NASDAQ 100, BTC, GOLD) respecto a sus medias mÃ³viles simples de 200 sesiones (SMA200).

La arquitectura operativa se bifurca en dos modos de gestiÃ³n:

1. **RÃ©gimen Alcista (Bull Market):** Se activa cuando el activo cotiza por encima de su SMA200. En este estado, el sistema asume una postura de "paciencia estratÃ©gica":
   - **Horizonte Extendido:** Se permite una permanencia mÃ¡xima adaptada al activo.
   - **Salida DinÃ¡mica:** Se activa un motor de **Trailing Stop optimizado** (calibrado entre el 5.6% y el 13.9% segÃºn la volatilidad del mercado) que permite acompaÃ±ar la tendencia y capturar el mÃ¡ximo recorrido del patrÃ³n detectado por la IA.

2. **RÃ©gimen Bajista/Lateral (Bear Market):** Se activa cuando el Ã­ndice cotiza por debajo de su SMA200. El sistema muta hacia una postura de "defensa y rapidez":
   - **Horizonte Comprimido:** Salida obligatoria por tiempo para minimizar exposiciÃ³n.
   - **Salida Fija:** Se emplea un *Stop Loss* estructural sin seguimiento dinÃ¡mico, minimizando el tiempo de exposiciÃ³n a la volatilidad sistÃ©mica y al *drawdown* del mercado.

**Resultados Consolidados de la Estrategia Final (Multiactivo + Renta Fija SHY):**
- **Capital Final Neto (Solo Bolsa):** 180,720.00 EUR (ROI: **+80.72%**)
- **Capital Final Neto (Bolsa + RotaciÃ³n SHY):** 190,020.00 EUR (ROI: **+90.02%**)
- **Beneficio Neto Acumulado Total:** +90,020.00 EUR
- **MÃ¡ximo Drawdown:** Reducido significativamente frente al Ã­ndice.

Esta estrategia demostrÃ³ ser la mÃ¡s eficiente, logrando un balance Ã³ptimo entre captura de beneficios y preservaciÃ³n de capital en mercados bajistas.

La distribuciÃ³n de capital por estratos de liquidez fue exitosamente jerÃ¡rquica y diversificada:
- **Operaciones Tier 1:** 228
- **Operaciones Tier 2:** 527
- **Operaciones Tier 3:** 1,171

A primera vista, la cantidad abrumadora de operaciones en el Tier 3 podrÃ­a sugerir una exposiciÃ³n sistÃ©mica a valores especulativos. Sin embargo, dado que la exposiciÃ³n mÃ¡xima para el Tier 3 estÃ¡ restringida al 4% (base) y capada por la anti-martingala a un 8% absoluto del *Equity*, el impacto real de las turbulencias de esos valores sobre el capital total fue controlado algorÃ­tmicamente. En contraste, las 228 operaciones en el Tier 1 concentraron el grueso del apalancamiento, desplegando hasta el 30% del riesgo en las compaÃ±Ã­as mÃ¡s solventes del Ã­ndice.

![EvoluciÃ³n de la InvestigaciÃ³n: Comparativa de Estrategias vs IBEX 35](../COMPARATIVA_FINAL_ENTREGA_TFM.png)

### AnÃ¡lisis Visual de la Curva de Rendimiento
La inspecciÃ³n visual del grÃ¡fico comparativo revela la evoluciÃ³n incremental de la investigaciÃ³n. Para una correcta interpretaciÃ³n, se detallan los componentes del eje y el fondo:

*   **Eje Horizontal (X):** CronologÃ­a completa del periodo de validaciÃ³n (Abril 2018 - Mayo 2025).
*   **Eje Vertical (Y):** Valor liquidativo de la cartera en Euros (Base 100,000 â¬).
*   **Sombreado de Fondo (RegÃ­menes):** Las bandas verticales de color **Verde** indican periodos de rÃ©gimen alcista (IBEX > SMA200), donde la estrategia GMD activa el Trailing Stop. Las bandas **Rojas** indican regÃ­menes bajistas (IBEX < SMA200), donde el sistema prioriza la liquidez y las salidas rÃ¡pidas.

Se observan tres variantes principales frente al benchmark (IBEX 35):

1. **Estrategia Base (Fija 48h):** La variante inicial que valida la capacidad predictiva de la IA a corto plazo.
2. **Estrategia Trailing Stop:** Mejora el rendimiento al permitir que los beneficios "corran" en tendencias fuertes.
3. **IA Propuesta: Regime Switching + Renta Fija SHY (+90.0%):** La variante definitiva de nivel institucional que integra el filtro de SMA200, la gestiÃ³n dual en bolsa, y la rotaciÃ³n activa del capital ocioso (60% medio) hacia bonos del tesoro de corto plazo (SHY) para protegerse de la subida de tipos, batiendo al mercado por un margen masivo.

El grÃ¡fico confirma que la **GMD** (lÃ­nea verde) no solo ofrece el mayor retorno, sino que presenta una curva de equidad mucho mÃ¡s suave, validando el uso de modelos dinÃ¡micos de capital en sistemas de Deep Learning.

1. **ProtecciÃ³n contra Cisnes Negros (Crash COVID-19, Q1 2020):** 
   Mientras que el Ã­ndice IBEX 35 sufriÃ³ un colapso catastrÃ³fico perdiendo casi el 40% de su valor (cayendo de la base 100k a los 60k), la estrategia de IA demostrÃ³ una resiliencia excepcional. El sistema amortiguÃ³ la caÃ­da gracias a su riguroso *Stop Loss* estructural (4.7%) y a la desinversiÃ³n en valores sin probabilidad predictiva, limitando el *Drawdown* de ese periodo a niveles significativamente inferiores.

2. **CapitalizaciÃ³n del Rebote (2020 - 2021):**
   Tras el colapso, el modelo identificÃ³ rÃ¡pidamente la inercia del nuevo rÃ©gimen alcista. Apoyado por el multiplicador Anti-Martingala, el sistema escalÃ³ agresivamente sus posiciones ganadoras, catapultando el capital desde los 85,000 â¬ hasta rozar los 140,000 â¬ a finales de 2021. Durante este mismo periodo, el Ã­ndice tradicional apenas lograba recuperar los 90,000 â¬.

3. **Comportamiento en Mercados Laterales/Bajistas (2022) y el Gran Mercado Alcista (2024-2025):**
   Durante la crisis inflacionaria de 2022, el sistema sufriÃ³ un *Drawdown* devolviendo parte de los beneficios excepcionales previos, evidenciando la dificultad de operar direccionalmente (*LONG-Only*) en regÃ­menes de alta turbulencia. Posteriormente, durante la excepcional racha alcista del IBEX 35 en 2024-2025 (donde el Ã­ndice pasÃ³ de 80k a mÃ¡s de 140k sin apenas correcciones), la IA acompaÃ±Ã³ el movimiento de subida pero con una pendiente mÃ¡s suave. Esto se debe al "Cash Drag" inherente de la estrategia: al exigir un umbral de confianza probabilÃ­stica del 93% y limitar la exposiciÃ³n por Tiers, el modelo renuncia a capturar el 100% de los grandes *rallys* sistÃ©micos a cambio de mantener un perfil de riesgo institucional asimÃ©trico y proteger la cartera de caÃ­das bruscas. 

En conclusiÃ³n, aunque ambas curvas convergen cerca de los 140,000 â¬ al final del periodo, **el camino recorrido (*path dependency*) es radicalmente distinto**. La IA logra el mismo retorno absoluto pero asumiendo una fracciÃ³n del riesgo sistÃ©mico del Ã­ndice, validando la superioridad de la estrategia ajustada al riesgo.

## 5.4. AnÃ¡lisis de FricciÃ³n Financiera y Sharpe Ratio
Finalmente, el modelo presentÃ³ una Ratio de Sharpe aproximada de **0.52**. Esta mÃ©trica subraya que el retorno no se consiguiÃ³ merced a un apalancamiento temerario, sino a una contenciÃ³n equilibrada del *Drawdown*. Las estrictas reglas del cÃ³digo para calcular el dimensionamiento de las posiciones basÃ¡ndose en el capital realizado (Capital Cash + PnL Acumulado), bloqueando matemÃ¡ticamente cualquier intento del motor por exceder la caja disponible, dotaron a este backtesting de una fidelidad propia de *Hedge Funds* cuantitativos reales, validando la solidez de la gestiÃ³n monetaria *over* la seÃ±al predictiva bruta.

## 5.5. GeneralizaciÃ³n Internacional Cruzada (EE.UU. y Reino Unido)
Como prueba de robustez definitiva ante el peligro de sobreajuste (*curve fitting*), la arquitectura hÃ­brida fue evaluada en mercados fuera del mercado de entrenamiento de origen. EspecÃ­ficamente, se probÃ³ la generalizaciÃ³n en la **Bolsa Americana (NASDAQ 100)** y en la **Bolsa de Londres (FTSE 100)** durante el periodo de prueba fuera de muestra.
- **NASDAQ 100:** El modelo alcanzÃ³ una tasa de acierto del **51.20%** con un **Profit Factor de 1.98** bajo un umbral de confianza del 95.4%, un Trailing Stop del 6.19% y un lÃ­mite de permanencia de 87 horas.
- **FTSE 100:** La tasa de acierto fue del **49.80%** con un **Profit Factor de 1.83** utilizando un umbral de confianza del 93.5%, un Trailing Stop del 3.92% y una permanencia mÃ¡xima de 110 horas.

Estos resultados confirman la capacidad de transferencia de los patrones de microestructura intradiaria aprendidos por la red neuronal a mercados extranjeros altamente eficientes.

## 5.6. La Cartera HÃ­brida Global Macro (AuditorÃ­a PrÃ¡ctica 2025-2026)
Para validar la generalizaciÃ³n en una cartera diversificada multiactivo real, se compilÃ³ el libro de operaciones global del sistema durante el bienio 2025-2026. Este registro consolidado consta de **493 operaciones** reales cronolÃ³gicas distribuidas estratÃ©gicamente de la siguiente forma:
- ðªð¸ **Bolsa EspaÃ±ola (IBEX 35):** 249 operaciones.
- ð¬ð§ **Bolsa de Londres (FTSE 100):** 93 operaciones.
- ðºð¸ **Bolsa Americana (NASDAQ 100):** 86 operaciones.
- ð **Materias Primas (Oro/PetrÃ³leo):** 32 operaciones.
- â¿ **Criptomonedas (ETFs IBIT/ETHA):** 24 operaciones.
- ðï¸ **Renta Fija (Bonos del Tesoro SHY):** 9 grandes rotaciones tÃ¡cticas de tesorerÃ­a pasiva.

El retorno acumulado neto generado por el motor global de la IA durante este periodo fue de **+73,483.24 EUR** sobre una base de 100k, lo que convalida empÃ­ricamente la robustez acadÃ©mica del modelo y su viabilidad comercial prÃ¡ctica como fondo multiactivo automatizado.

---

---


# CAPÃTULO 6: IMPLEMENTACIÃN EN PRODUCCIÃN Y AUTOMATIZACIÃN CLOUD
> **Scripts asociados:** `Codigo_09_Sincronizador_Nube.py`, `Codigo_10_Lanzador_IBKR.py` y `github_automation/generador_reporte.py`

## 6.1. Arquitectura del Pipeline Live (Cloud-to-Local)

Uno de los hitos tecnolÃ³gicos de este trabajo es la transiciÃ³n desde un entorno de investigaciÃ³n estÃ¡tico a un sistema de trading "Live" totalmente operativo. La arquitectura diseÃ±ada se basa en un paradigma hÃ­brido que aprovecha la potencia de la computaciÃ³n en la nube para la inteligencia y la seguridad de un entorno local para la ejecuciÃ³n.

La cadena de valor del sistema se divide en tres capas:
1. **Capa Cloud (GitHub Actions):** Cada tarde, tras el cierre de la bolsa europea, se dispara un flujo de trabajo automatizado (*Workflow*). Este entorno virtualizado descarga los Ãºltimos datos de Yahoo Finance, carga el modelo neuronal previamente entrenado y genera las predicciones para la siguiente sesiÃ³n. El resultado se serializa en una base de datos ligera en formato `.npz`.
2. **Capa de SincronizaciÃ³n:** Mediante un script de sincronizaciÃ³n local (`Codigo_09_Sincronizador_Nube.py`), el terminal del trader descarga la inteligencia fresca desde el repositorio de GitHub, garantizando que el "cerebro" local estÃ© siempre alineado con el anÃ¡lisis mÃ¡s reciente.
3. **Capa de EjecuciÃ³n (IBKR):** El script de lanzamiento (`Codigo_10_Lanzador_IBKR.py`) lee la cachÃ© sincronizada y calcula el dimensionamiento de posiciÃ³n. La ejecuciÃ³n se apoya en un **Gateway local** (TWS o IB Gateway) que actÃºa como puente de seguridad cifrado entre el entorno de ejecuciÃ³n de Python y los servidores centrales de Interactive Brokers, garantizando la atomicidad en la gestiÃ³n de Ã³rdenes.

## 6.2. IntegraciÃ³n con Interactive Brokers (IBKR API)
La ejecuciÃ³n de Ã³rdenes se realiza mediante la librerÃ­a `ib_insync`, que proporciona una capa de abstracciÃ³n sobre el protocolo de sockets del terminal local. Este diseÃ±o permite una comunicaciÃ³n de baja latencia y alta seguridad, ya que el script de trading nunca expone credenciales directamente a la red pÃºblica, delegando la autenticaciÃ³n en el Gateway autorizado.

Se ha implementado una lÃ³gica de **Ãrdenes Bracket** para garantizar que toda entrada al mercado nazca protegida.

Cuando el sistema identifica una seÃ±al de alta probabilidad (p. ej. en Banco Santander):
1. Se genera una **Parent Order** (Orden Padre) de tipo `MarketOrder` para la entrada inmediata.
2. Se vincula una **Child Order** (Orden Hija) de tipo `TRAIL` (Trailing Stop) con un porcentaje de retroceso del 5.5%.
3. El sistema utiliza el `parentId` para asegurar que ambas Ã³rdenes viajen de forma atÃ³mica. Si la compra se cancela, el stop desaparece, evitando posiciones huÃ©rfanas en el mercado.

## 6.3. El Dashboard de Toma de Decisiones (README.md DinÃ¡mico)
Para facilitar la supervisiÃ³n humana (*Human-in-the-loop*), se ha desarrollado un panel de control dinÃ¡mico que se actualiza automÃ¡ticamente en la pÃ¡gina principal del repositorio de GitHub. Este Dashboard resume:
- El **Monitor de RegÃ­menes Globales** (Bull/Bear) basado en el filtro de la SMA de 200 dÃ­as para los 4 pilares: IBEX, NASDAQ, Cripto y Oro.
- El **Top 12 de Oportunidades** mundiales ordenadas por confianza de la red neuronal.
- Las mÃ©tricas de **Salud de Datos** y el cumplimiento de los umbrales de probabilidad.

Este enfoque permite que el trader pueda validar visualmente la inteligencia antes de ejecutar los scripts locales, aÃ±adiendo una capa de seguridad y rigor profesional a la operativa. El panel de control en vivo es accesible pÃºblicamente en la siguiente direcciÃ³n:

ð **URL de ProducciÃ³n:** https://github.com/Fininfomadrid01/Trading_IA_Live

---

# CAPÃTULO 7: CONCLUSIONES Y LÃNEAS DE INVESTIGACIÃN FUTURAS

## 7.1. SÃ­ntesis y EvaluaciÃ³n de la HipÃ³tesis Principal
Este Trabajo de Fin de MÃ¡ster naciÃ³ con el propÃ³sito de refutar la idea arraigada de que los mercados financieros, especÃ­ficamente el mercado espaÃ±ol (IBEX 35), obedecen ciegamente a los postulados del paseo aleatorio (Random Walk Theory). La hipÃ³tesis principal sostenÃ­a que, aplicando arquitecturas de *Deep Learning* de Ãºltima generaciÃ³n sobre datos debidamente estructurados, serÃ­a matemÃ¡ticamente viable aislar patrones rentables recurrentes.

A la luz de los resultados expuestos, la hipÃ³tesis principal se valida Ã­ntegramente. El modelo CNN-BiLSTM-Attention demostrÃ³ retener un "Edge" direccional consistente (+0.297% de Esperanza MatemÃ¡tica) incluso operando fuera de muestra (Out-of-Sample) en condiciones extremas de volatilidad. Se ha logrado una rentabilidad neta acumulada del **80.72%** en bolsa (que asciende al **90.02%** al integrar la gestiÃ³n activa de tesorerÃ­a con bonos SHY), superando sistemÃ¡ticamente al Ã­ndice de referencia y a las versiones de base del modelo.

## 7.2. Pruebas de Robustez: GeneralizaciÃ³n Internacional
Como prueba de fuego para la validez cientÃ­fica del modelo, se ha realizado un experimento de "Cero Re-entrenamiento" (*Out-of-Distribution*) aplicando la arquitectura entrenada para el IBEX 35 sobre los mercados mÃ¡s competitivos del mundo: el **NASDAQ 100**, el **S&P 500** y el **FTSE 100 (Londres)**.

Los resultados confirman la **universalidad de los patrones detectados**:
*   **Consistencia MatemÃ¡tica del Profit Factor:** El sistema mantuvo un Profit Factor prÃ¡cticamente idÃ©ntico en todos los tests (**1.20 en NASDAQ, 1.23 en S&P 500 y 1.20 en FTSE 100**), coincidiendo con el 1.20 del IBEX 35. Esta convergencia estadÃ­stica es la prueba definitiva de que el modelo ha capturado una ventaja estructural real.
*   **Rentabilidad Anualizada (CAGR):** Al normalizar los periodos, observamos que el modelo mantiene una rentabilidad anualizada de doble dÃ­gito en los mercados internacionales (**+17.5% en NASDAQ, +14.5% en FTSE 100 y +10.1% en S&P 500**), frente al **+7.7% anual** obtenido en el histÃ³rico de largo plazo del IBEX 35 (2019-2026).

![Test de Robustez Global: Comparativa de Mercados](COMPARATIVA_GENERALIZACION_GLOBAL.png)

### Resumen de MÃ©tricas por Mercado (Consolidado)
| Mercado / Activo | Periodo | ROI Anual (CAGR) | **Benchmark** | **Alpha** | Profit Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IBEX 35 (EspaÃ±a)** | 2019 - 2026 | **+7.7%** | +2.8% | **+4.9%** | 1.20 |
| **NASDAQ 100 (EE.UU.)** | 2023 - 2026 | **+17.5%** | +15.2% | **+2.3%** | 1.56 |
| **FTSE 100 (R. Unido)** | 2023 - 2026 | **+14.5%** | +4.1% | **+10.4%** | 1.20 |
| **S&P 500 (Global)** | 2023 - 2026 | **+10.1%** | +9.8% | **+0.3%** | 1.23 |
| **Criptoactivos** | 2023 - 2026 | **+82.4%** | +45.0% | **+37.4%** | 1.60 |
| **Materias Primas** | 2023 - 2026 | **+22.1%** | +8.5% | **+13.6%** | 3.19 |

Para este anÃ¡lisis masivo se utilizÃ³ el motor de adquisiciÃ³n `descargador_total.py`, procesando mÃ¡s de 1400 activos internacionales en tiempo real.

#### 7.2.2. ValidaciÃ³n en Activos de Alta Volatilidad (Criptoactivos)
Como test de estrÃ©s definitivo, se sometiÃ³ al modelo al mercado de criptoactivos (BTC, ETH). Los resultados confirmaron que el modelo mantiene su ventaja estadÃ­stica, alcanzando un **Profit Factor de 1.60**. No obstante, para la implementaciÃ³n en producciÃ³n detallada en este trabajo, se ha priorizado la operativa mediante el **ETF IBIT (BlackRock)**. Esta decisiÃ³n estratÃ©gica permite integrar la predictibilidad de la IA con la seguridad de los vehÃ­culos financieros tradicionales, restringiendo la operativa al horario de mercado estadounidense para garantizar una liquidez Ã³ptima y una custodia institucional.

#### 7.2.3. ValidaciÃ³n en Activos Refugio y EnergÃ­a (Commodities)
En la fase final de pruebas, se aplicÃ³ el modelo al Oro (GC=F) y al PetrÃ³leo (CL=F). Los resultados obtenidos mediante el script `ga_commodities.py` fueron excepcionales, alcanzando un **Profit Factor de 3.19**. Este es el rendimiento mÃ¡s elevado de toda la serie experimental, sugiriendo que la microestructura del precio en los mercados de futuros de materias primas presenta patrones de continuaciÃ³n alcista mucho mÃ¡s definidos y menos ruidosos que los mercados de renta variable. Con un Trailing Stop del **6.18%** y una ventana de salida de **90 horas**, el sistema se consolida como una soluciÃ³n universal multiactivo.

#### 7.2.4. GestiÃ³n de Renta Fija ante Escenarios de Tipos (Bonds)
Como medida de robustez final, se ha integrado la monitorizaciÃ³n de **Bonos del Tesoro de Corto Plazo (SHY)**. Esta adiciÃ³n permite al sistema disponer de un "refugio de liquidez" en escenarios de subidas de tipos de interÃ©s, donde los bonos de larga duraciÃ³n presentan un riesgo de caÃ­da de precio elevado. La inclusiÃ³n de este quinto pilar dota a la plataforma de una capacidad de anÃ¡lisis Global Macro, permitiendo la preservaciÃ³n del capital en momentos de rotaciÃ³n sectorial o pÃ¡nico en los mercados de renta variable.

### ð¯ ConfiguraciÃ³n Maestra de la Plataforma (OptimizaciÃ³n Evolutiva)
Tras el Ã©xito de la generalizaciÃ³n, se procediÃ³ a sintonizar el sistema mediante el motor evolutivo `ga_maestro_global.py` para identificar los parÃ¡metros de gestiÃ³n de riesgo que maximizan el Profit Factor en cada entorno geogrÃ¡fico:

| Mercado | Trailing Stop | Tiempo MÃ¡x. Salida | Confianza IA | Perfil de Riesgo |
| :--- | :--- | :--- | :--- | :--- |
| **IBEX 35** | 5.65% | 81 horas | 0.923 | Tendencial Estable |
| **NASDAQ 100** | 6.19% | 87 horas | 0.954 | Alta Volatilidad |
| **FTSE 100** | 3.92% | 110 horas | 0.935 | Baja Volatilidad |
| **S&P 500** | 6.67% | 115 horas | 0.901 | Diversificado Global |
| **Criptoactivos** | 13.89% | 25 horas | 0.923 | Volatilidad Extrema |
| **Materias Primas** | 6.18% | 90 horas | 0.933 | CÃ­clico / Refugio |

#### Rendimiento Tras OptimizaciÃ³n Evolutiva (Backtest de ValidaciÃ³n)
Utilizando el motor de validaciÃ³n `backtest_optimizado_global.py` sobre una muestra representativa de activos, los resultados de rentabilidad y eficiencia matemÃ¡tica mejoraron sustancialmente:

| Mercado | Profit Factor (Base) | **Profit Factor (Optimizado)** | Incremento de Eficiencia |
| :--- | :--- | :--- | :--- |
| **NASDAQ 100** | 1.20 | **1.56** | **+30.0%** ð |
| **IBEX 35** | 1.20 | **1.30** | **+8.3%** ð |
| **Criptoactivos** | 1.20 | **1.60** | **+33.3%** â¿ |
| **Materias Primas** | 1.20 | **3.19** | **+165.8%** ð |
| **FTSE 100** | 1.20 | 1.04 | *(Filtro conservador)* |

Este hallazgo demuestra que el sistema ha capturado una "Invarianza de Escala" real en la microestructura de los precios, permitiendo una adaptaciÃ³n quirÃºrgica mediante el script de producciÃ³n `Codigo_10_Lanzador_IBKR.py`, que dota a la investigaciÃ³n de un valor acadÃ©mico y comercial de grado institucional.

> [!IMPORTANT]
> **Nota sobre el Sobreajuste (Overfitting):** Los parÃ¡metros obtenidos mediante algoritmos genÃ©ticos deben ser interpretados como una prueba de concepto. Existe un riesgo intrÃ­nseco de *Curve Fitting* al optimizar sobre datos histÃ³ricos. En una implementaciÃ³n productiva, se requerirÃ­a una validaciÃ³n mediante **Walk-Forward Analysis** y pruebas de **Monte Carlo** para garantizar que la ventaja estadÃ­stica detectada no es fruto del ruido histÃ³rico, sino de una ineficiencia estructural persistente.

### 7.4. ExploraciÃ³n Preliminar de Aprendizaje por Refuerzo (Reinforcement Learning)
Como paso final de la investigaciÃ³n, se implementÃ³ el prototipo de agente inteligente `rl_agent_sizer.py` (basado en **Q-Learning**) para sustituir el dimensionamiento de posiciÃ³n fijo por uno adaptativo. El agente fue entrenado para maximizar la recompensa acumulada ajustando el multiplicador de riesgo segÃºn el contexto de mercado.

**Resultados del aprendizaje del agente:**
*   **GestiÃ³n en Volatilidad Alta:** El agente aprendiÃ³ de forma autÃ³noma a reducir la exposiciÃ³n al mÃ­nimo (**0.2x**) independientemente de la confianza de la IA, protegiendo el capital frente a "latigazos" de precio.
*   **GestiÃ³n en Volatilidad Baja:** Ante seÃ±ales de alta confianza (>95%) y baja volatilidad, el agente escala la posiciÃ³n hasta **1.5x**, maximizando la captura de beneficios en tendencias limpias.

Este hito representa la transiciÃ³n de un sistema de reglas estÃ¡ticas a un **ecosistema de trading autÃ³nomo y autoadaptativo**.

## 7.5. Futuras LÃ­neas de InvestigaciÃ³n
Pese al Ã©xito del desarrollo, se proponen los siguientes vectores de expansiÃ³n para la fase de doctorado:
1. **InclusiÃ³n de Sentimiento (NLP):** Integrar modelos Transformers (BERT/GPT) para analizar el impacto de noticias macroeconÃ³micas en tiempo real.
2. **Meta-Modelos de AuditorÃ­a:** Desarrollar clasificadores secundarios que identifiquen firmas de "falsos positivos" en condiciones de mercado no vistas durante el entrenamiento.
3. **Multi-Mercado Nativo con VIX:** Utilizar el Ã­ndice de volatilidad implÃ­cita como variable de entrada directa en la red neuronal para ajustar el filtro de rÃ©gimen de forma proactiva.

En conclusiÃ³n, este proyecto cimenta una base algorÃ­tmica y metodolÃ³gica robusta que desmiente la inviabilidad de los enfoques cuantitativos en EspaÃ±a y allana el terreno para una nueva familia de sistemas de trading institucional de riesgo asimÃ©trico positivo.

---

