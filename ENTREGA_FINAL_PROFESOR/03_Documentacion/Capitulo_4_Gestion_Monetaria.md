# CAPÍTULO 4: MOTOR DE EJECUCIÓN Y GESTIÓN MONETARIA AVANZADA

## 4.1. El Problema Clásico del Dimensionamiento de Capital
Uno de los fallos metodológicos más recurrentes en la literatura académica de predicción bursátil es asumir liquidez infinita y capital inagotable. Es común que los modelos teóricos simulen inversiones idénticas (e.g., apostar 10.000€ a cada señal producida) sin contemplar si en ese momento temporal el sistema posee efectivo disponible, o si el activo subyacente soporta ese nivel de absorción de liquidez sin generar *slippage* severo.

Para alinear el presente modelo predictivo con la rigurosidad institucional y académica requerida, se ha desarrollado un Motor de Gestión Monetaria de "Fase 2" (*Phase 2: Money Management Engine*). Este motor impone una estricta separación metodológica entre el **Plano de la Señal** (cuántas veces acierta la IA y con qué rentabilidad pura) y el **Plano de la Cartera** (cómo se despliega de forma inteligente y jerárquica el capital finito de la cuenta para protegerse de la quiebra).

## 4.2. Filtrado de Restricción Estructural y Esperanza Matemática
Antes de arriesgar capital, toda señal de la inteligencia artificial debe ser evaluada probabilísticamente. La Esperanza Matemática ($EV$) mide el retorno real esperado por operación integrando tanto la precisión heurística como el Ratio Riesgo-Beneficio:

$$ EV = (WR \times AvgWin) + ((1 - WR) \times AvgLoss) $$

Adicionalmente, se programó una Restricción de Riesgo Estructural (*Structural Risk Filter*). Un algoritmo genético previo calculó que el riesgo óptimo de ruina para este mercado exigía una volatilidad base de invalidación (`min_struct_risk`) del 4.7%. Por consiguiente, el motor rechaza cualquier señal que no permita establecer matemáticamente un umbral de stop loss al -4.7% de distancia, purificando la cartera de operaciones con un perfil de riesgo asimétrico desfavorable.

## 4.3. Agrupamiento por Tiers de Liquidez (Lagged Rebalancing)
El IBEX 35 es extremadamente asimétrico en términos de volumen de negociación. Invertir el mismo porcentaje de capital en el Banco Santander (alto volumen) que en Sacyr (bajo volumen) expone a la cartera a riesgos de liquidez catastróficos. 

Para resolver este problema de manera limpia y sin incurrir en sesgo de conocimiento a futuro (*look-ahead bias*), el motor de capital ejecuta una ponderación dinámica trimestral:
1. Al inicio de cada trimestre, el algoritmo analiza el volumen mediano negociado por cada *ticker* **durante el trimestre anterior**.
2. Los valores se ordenan de mayor a menor liquidez histórica y se fraccionan matemáticamente en tres tercios (*Tiers*).
3. La cuenta asigna presupuestos fijos de exposición en base a este ranking:
   - **Tier 1 (Alta Liquidez):** Exposición base del **15%** de la Equidad Total.
   - **Tier 2 (Media Liquidez):** Exposición base del **8%**.
   - **Tier 3 (Baja Liquidez):** Exposición base del **4%**.

Esta arquitectura restringe dramáticamente el riesgo sistémico de los chicharros, forzando a la cartera a priorizar de forma algorítmica las ejecuciones institucionales.

## 4.4. Generalización Internacional: Presupuestación de Riesgo (Risk Budgeting)
Aunque en los mercados internacionales (NASDAQ 100, FTSE 100, Criptomonedas y Futuros sobre Commodities) no exista un riesgo real de *slippage* debido a su colosal volumen y profundidad diaria —haciendo innecesaria la restricción de liquidez por Tiers—, sí resulta obligatorio limitar el capital asignado en base a su perfil de **Volatilidad y Correlación**. En finanzas cuantitativas profesionales, esta estrategia de control se denomina **Presupuestación de Riesgo (Risk Budgeting)**.

Si el sistema tratase a todos los activos por igual asignándoles un presupuesto máximo (Tier 1 del 15%), la cartera global sufriría desplomes (*drawdowns*) extremos e inaceptables cuando los activos de alta volatilidad (como el Bitcoin o el Petróleo) experimentasen correcciones bruscas. Por consiguiente, el motor global macro de la plataforma asigna de forma nativa equivalencias de Tiers ajustadas a la volatilidad histórica de cada tipo de activo:
*   **Equivalencia a Tier 3 (Exposición Base del 4% - 5%):** Se aplica a **Criptoactivos (ETFs IBIT / ETHA)**. No es por falta de liquidez (poseen profundidad de sobra), sino para neutralizar su volatilidad extrema. Limitar la posición al 4% actúa como un amortiguador, garantizando que un latigazo del -15% en el Bitcoin no altere de forma catastrófica la curva de capital de la cuenta.
*   **Equivalencia a Tier 2 (Exposición Base del 8%):** Se aplica a **Materias Primas (Oro/Petróleo)** y a la **Bolsa de Londres (FTSE 100)**. Ofrecen una volatilidad media-baja y actúan como excelentes diversificadores globales o de activos reales.
*   **Equivalencia a Tier 1 (Exposición Base del 10% - 15%):** Se aplica a las **Blue Chips del NASDAQ 100 e IBEX 35**. Constituyen el motor de crecimiento y el pilar direccional de renta variable principal de la cartera.

Esta sofisticación dota al sistema de una robustez matemática de grado institucional, permitiendo su despliegue y preservación de capital multiactivo.

## 4.5. Teoría Anti-Martingala Calibrada
El despliegue estático de capital es subóptimo, ya que ignora el momento probabilístico de la serie. Para maximizar el efecto del interés compuesto sin incrementar el riesgo de ruina, se integró una política de **Anti-Martingala Calibrada**. 

A diferencia de las estrategias suicidas de Martingala (que doblan la apuesta al perder), la Anti-Martingala protege el capital en periodos de ruido y lo acelera en periodos de "Edge" direccional:
- Se asigna un Multiplicador de Riesgo $M_T$ (iniciado en 1.0) independiente a cada Tier.
- **Si el trade resulta ganador (TP):** $M_T = M_T \times 1.17$
- **Si el trade resulta perdedor (SL):** $M_T = M_T \times 0.96$
- Para evitar sobreexposiciones letales, el multiplicador está acotado estrictamente matemáticamente en el dominio continuo de $[0.50, 2.00]$.

Por tanto, una operación del Tier 1 puede expandir su riesgo dinámicamente hasta comprometer el 30% del capital si, y solo si, la inteligencia artificial está atravesando una racha estadística excepcionalmente predecible.

## 4.6. Resolución del "Cash Drag" e Integridad Contable
Finalmente, el modelo aborda el dilema clásico del *Cash Drag* (capital no invertido) bajo la restricción del efectivo. En situaciones donde múltiples señales coinciden en el mismo día temporal, el modelo simula el tiempo exacto. Ordena las oportunidades priorizando **estrictamente por la Probabilidad emitida por la CNN**, calculando el tamaño ($Size = Equidad \times Tier \times M_T$). Si la "Caja Restante" es matemáticamente inferior al tamaño requerido, el simulador institucional rechaza la operación instantáneamente, replicando la escasez de margen real y validando su credibilidad empírica total.
