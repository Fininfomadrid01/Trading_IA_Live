# CONFIGURACIÓN MAESTRA DE LA PLATAFORMA (INVESTIGACIÓN AVANZADA)

Este documento detalla los parámetros óptimos de gestión de riesgo y ejecución para el modelo CNN-BiLSTM-Attention, obtenidos mediante optimización evolutiva (Algoritmos Genéticos).

## 1. Parámetros Específicos por Mercado

| Mercado | Símbolo Ref. | Trailing Stop | Max Time-Exit | Umbral de Confianza |
| :--- | :--- | :--- | :--- | :--- |
| **IBEX 35** | `^IBEX` | 5.65% | 81 horas | 0.923 |
| **NASDAQ 100** | `^NDX` | 6.19% | 87 horas | 0.954 |
| **FTSE 100** | `^FTSE` | 3.92% | 110 horas | 0.935 |
| **S&P 500** | `SPY` | 6.67% | 115 horas | 0.901 |

## 2. Lógica de Ejecución Recomendada
Para maximizar el Profit Factor observado en los experimentos (1.47 - 1.86), se recomienda:
1. **Filtro de Régimen:** Ejecutar solo operaciones en sentido de la SMA 200 del índice de referencia.
2. **Trailing Dinámico:** El stop debe seguir al precio máximo alcanzado desde la entrada (`max_high * (1 - trailing_pct)`).
3. **Cierre por Tiempo:** Si tras el periodo de `Max Time-Exit` la operación sigue abierta, se debe cerrar a mercado para liberar capital, independientemente del PnL.

## 3. Conclusiones de I+D
La optimización genética demuestra que una configuración única (*One-size-fits-all*) suboptimiza el rendimiento global. La adaptación a la microestructura de cada mercado (volatilidad media y velocidad de tendencia) permite incrementar la Esperanza Matemática del sistema de forma significativa.
