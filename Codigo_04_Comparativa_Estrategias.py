import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import matplotlib.ticker as ticker

# Rutas de archivos
original_file = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_MM_Avanzada.csv"
advanced_file = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_Avanzados.csv"
out_img = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\Comparativa_Estrategias_TFM.png"

def get_equity_curve(file_path, capital_inicial=100000.0):
    df = pd.read_csv(file_path)
    df['Exit_Date'] = pd.to_datetime(df['exit_ts']).dt.floor('d')
    df_equity = df.groupby('Exit_Date')['pnl_neto'].sum().reset_index()
    df_equity = df_equity.sort_values('Exit_Date')
    
    fechas_rango = pd.date_range(start='2018-01-02', end='2025-06-11', freq='B')
    equity_series = pd.Series(0, index=fechas_rango, dtype=float)
    
    for _, row in df_equity.iterrows():
        if row['Exit_Date'] in equity_series.index:
            equity_series[row['Exit_Date']] += row['pnl_neto']
    
    balance = capital_inicial + equity_series.cumsum()
    return balance.ffill().fillna(capital_inicial)

# Cargar curvas
balance_original = get_equity_curve(original_file)
balance_avanzado = get_equity_curve(advanced_file)

# Cargar IBEX
ibex_oficial = yf.download('^IBEX', start='2018-01-02', end='2025-06-12', progress=False)
ibex_close = ibex_oficial['Close']['^IBEX'].reindex(balance_original.index).ffill()
if pd.isna(ibex_close.iloc[0]):
    first_valid = ibex_close.first_valid_index()
    if first_valid: ibex_close[:first_valid] = ibex_close[first_valid]

precio_inicial = float(ibex_close.iloc[0])
curva_ibex = (ibex_close / precio_inicial) * 100000.0

# Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 7))

# Líneas principales
ax.plot(balance_avanzado.index, balance_avanzado.values, color='#00ffcc', linewidth=2.5, label='Propuesta Final (Trailing Stop 5.5%)')
ax.plot(balance_original.index, balance_original.values, color='#ff3333', linewidth=2.0, alpha=0.7, label='Base TFM (Límite Fijo 48h)')
ax.plot(curva_ibex.index, curva_ibex.values, color='#ffffff', linewidth=1.5, linestyle='--', alpha=0.5, label='Benchmark IBEX 35')

# Sombreado
ax.fill_between(balance_avanzado.index, balance_avanzado.values, 100000, color='#00ffcc', alpha=0.05)

# Formato
formatter = ticker.StrMethodFormatter('{x:,.0f} €')
ax.yaxis.set_major_formatter(formatter)

ax.set_title('COMPARATIVA DE ESTRATEGIAS: EVOLUCIÓN DEL CAPITAL (2018-2025)', fontsize=16, pad=20, fontweight='bold')
ax.set_ylabel('Valor de la Cartera (EUR)', fontsize=12)
ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

# Formato de fechas
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=0)

# Anotaciones de métricas finales
final_adv = balance_avanzado.iloc[-1]
final_orig = balance_original.iloc[-1]
ax.text(balance_avanzado.index[-1], final_adv, f'  {final_adv:,.0f}€', color='#00ffcc', fontweight='bold', va='center')
ax.text(balance_original.index[-1], final_orig, f'  {final_orig:,.0f}€', color='#ff3333', fontweight='bold', va='center')

plt.tight_layout()
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Comparativa generada en: {out_img}")
