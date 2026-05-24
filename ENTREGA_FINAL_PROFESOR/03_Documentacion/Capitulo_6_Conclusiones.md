# CAPÍTULO 6: CONCLUSIONES Y TRABAJO FUTURO

## 6.1. Conclusiones sobre la Hipótesis de Investigación
La presente investigación ha demostrado con éxito que la integración de arquitecturas de *Deep Learning* (CNN-BiLSTM con Atención) y motores de Gestión Monetaria avanzada permite batir al índice de referencia IBEX 35 en términos de rentabilidad ajustada al riesgo. 

Los resultados finales, tras la implementación de la técnica de **Regime Switching** optimizada y diversificada, arrojan un ROI acumulado del **+80.72%** en bolsa. Al integrar además la estrategia de **Rotación Activa de Tesorería en Renta Fija (SHY)** para el capital no expuesto en bolsa (60% medio), el ROI final acumulado se eleva hasta un espectacular **+90.02%**, frente al **+46.23%** de la estrategia base. Este incremento excepcional en la rentabilidad absoluta no se ha logrado asumiendo más riesgo direccional, sino optimizando la tesorería pasiva mediante activos de bajísima duración inmunes a la subida de tipos de interés.

## 6.2. Hallazgos Cardinales
1.  **Superioridad de la Gestión Dinámica:** Se ha verificado que una señal predictiva con un *Win Rate* modesto (~40%) puede ser altamente rentable si se acompaña de una gestión de salidas asimétrica. La transición de salidas fijas (48h) a salidas dinámicas (Trailing Stop de 5.5% en mercados alcistas) ha elevado el **Profit Factor hasta 1.20**.
2.  **Mitigación del Riesgo Estructural:** El motor de *Tiers* de Liquidez ha sido fundamental para dotar al sistema de un perfil institucional, evitando la concentración de capital en valores ilíquidos y protegiendo la cartera durante eventos de "cisne negro" como el colapso de la COVID-19.
3.  **Filtrado de Régimen:** La observación de que las tendencias largas requieren "paciencia" (Trailing Stop) mientras que los mercados bajistas requieren "agilidad" (Cierre fijo) constituye el hallazgo técnico más relevante de este TFM. Esta combinación ha permitido reducir el *Drawdown* máximo al **27.79%**.

## 6.3. Limitaciones del Estudio
A pesar de los resultados positivos, el sistema presenta áreas de mejora:
*   **Fricción en Operativa Corta:** Se descartó la operativa de *Short Selling* en la propuesta final debido a que los costes de préstamo y la volatilidad inversa penalizaban excesivamente la esperanza matemática del modelo actual.
*   **Dependencia del Ticker:** El sistema muestra una alta sensibilidad a la calidad del dato histórico ajustado, subrayando la importancia crítica de la fase de ETL y corrección de *Corporate Actions*.

## 6.4. Líneas de Trabajo Futuro
Para futuras iteraciones de este sistema de trading algorítmico, se proponen las siguientes vías:
1.  **Análisis de Sentimiento (NLP):** Integrar una capa de procesamiento de noticias de prensa económica española (ej. Expansión, El Economista) para filtrar señales en momentos de alta incertidumbre geopolítica.
2.  **Optimización de Hiperparámetros Dinámica:** Implementar un motor de búsqueda bayesiana que ajuste el umbral de probabilidad y el porcentaje del Trailing Stop en tiempo real según la volatilidad implícita (VIX).
3.  **Inclusión de Futuros y ETFs:** Expandir el universo de activos hacia el futuro del IBEX 35 y ETFs sectoriales para mejorar la liquidez de la operativa en grandes capitales.

En definitiva, este trabajo constituye una base sólida para el desarrollo de sistemas de inversión semi-automáticos, validando la tesis de que la Inteligencia Artificial aplicada a la microestructura del precio es una herramienta poderosa para el gestor cuantitativo moderno.
