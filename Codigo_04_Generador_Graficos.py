import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import os

# 1. Cargar el log de trades de la IA
trades_file = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_MM_Avanzada.csv"
df_trades = pd.read_csv(trades_file)
df_trades['Exit_Date'] = pd.to_datetime(df_trades['exit_ts']).dt.floor('d')

# Ordenar por fecha de salida para construir la curva de equity
df_equity = df_trades.groupby('Exit_Date')['pnl_neto'].sum().reset_index()
df_equity = df_equity.sort_values('Exit_Date')

# Interpolar balance diario
capital_inicial = 100000.0
fechas_rango = pd.date_range(start='2018-01-02', end='2025-06-11', freq='B') # Business days
equity_series = pd.Series(index=fechas_rango, dtype=float).fillna(0)

for _, row in df_equity.iterrows():
    if row['Exit_Date'] in equity_series.index:
        equity_series[row['Exit_Date']] += row['pnl_neto']

balance_ia = capital_inicial + equity_series.cumsum()
balance_ia = balance_ia.ffill().fillna(capital_inicial)

# 2. Descargar curva OFICIAL del IBEX 35 con yfinance
print("Descargando datos oficiales del IBEX 35 (^IBEX)...")
ibex_oficial = yf.download('^IBEX', start='2018-01-02', end='2025-06-12', progress=False)

# Usaremos la columna Close
ibex_close = ibex_oficial['Close']['^IBEX']

# Reindexar para asegurar que tenemos las mismas fechas que la IA
ibex_close = ibex_close.reindex(balance_ia.index).ffill()
# Rellenar cualquier NaN inicial (por si el mercado estaba cerrado el 02-Ene pero BME no, o viceversa)
if pd.isna(ibex_close.iloc[0]):
    first_valid = ibex_close.first_valid_index()
    if first_valid:
        ibex_close[:first_valid] = ibex_close[first_valid]

# Crear base 100,000 para la comparativa
precio_inicial = float(ibex_close.iloc[0])
curva_ibex_oficial = (ibex_close / precio_inicial) * capital_inicial

print(f"Puntos IBEX Inicial: {precio_inicial:.2f}")
print(f"Puntos IBEX Final: {ibex_close.iloc[-1]:.2f}")

# 3. Dibujar el Gráfico
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(balance_ia.index, balance_ia.values, color='#00ffcc', linewidth=2.5, label='Estrategia IA (LONG-Only)')
ax.plot(curva_ibex_oficial.index, curva_ibex_oficial.values, color='#ffcc00', linewidth=2.0, alpha=0.8, label='Índice IBEX 35 Oficial (Yahoo Finance)')

ax.fill_between(balance_ia.index, balance_ia.values, 100000, color='#00ffcc', alpha=0.1)

import matplotlib.ticker as ticker
formatter = ticker.StrMethodFormatter('{x:,.0f} €')
ax.yaxis.set_major_formatter(formatter)

ax.set_title('Rendimiento OOS: Estrategia IA vs Índice IBEX 35 Oficial', fontsize=16, pad=15)
ax.set_ylabel('Capital (Euros)', fontsize=12)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
ax.legend(loc='upper left', fontsize=12)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar la imagen en AMBOS sitios (Desktop y Artifacts)
img_name = "Estrategia_vs_IBEX35_Real.png"
out_desktop = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\Estrategia_TFM_Final_vs_IBEX.png"


plt.savefig(out_desktop, dpi=300, bbox_inches='tight')

print("Generado usando datos OFICIALES del IBEX 35.")
