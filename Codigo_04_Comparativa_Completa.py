import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import matplotlib.ticker as ticker

# Rutas de archivos
original_file = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_MM_Avanzada.csv"
trailing_file = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_Trailing.csv"
regime_file   = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_Regime.csv"
out_img       = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\Comparativa_TODAS_Estrategias.png"

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
balance_trailing = get_equity_curve(trailing_file)
balance_regime   = get_equity_curve(regime_file)

# Cargar IBEX
ibex_oficial = yf.download('^IBEX', start='2017-01-01', end='2025-06-12', progress=False)
if isinstance(ibex_oficial.columns, pd.MultiIndex):
    ibex_close = ibex_oficial['Close']['^IBEX']
else:
    ibex_close = ibex_oficial['Close']

# Calcular Régimen (SMA200)
sma200 = ibex_close.rolling(window=200).mean()
ibex_regime = ibex_close > sma200

# Reindexar para el gráfico
ibex_close_plot = ibex_close.reindex(balance_original.index).ffill()
if pd.isna(ibex_close_plot.iloc[0]):
    first_valid = ibex_close_plot.first_valid_index()
    if first_valid: ibex_close_plot[:first_valid] = ibex_close_plot[first_valid]

precio_inicial = float(ibex_close_plot.iloc[0])
curva_ibex = (ibex_close_plot / precio_inicial) * 100000.0

# Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 8))

# Identificar cambios de régimen para las líneas verticales
ibex_regime_clean = ibex_regime.dropna()
switches = ibex_regime_clean.astype(int).diff().fillna(0)
switch_dates = switches[switches != 0].index

# Dibujar zonas de régimen en el fondo
first_date = balance_regime.index[0]
last_date  = balance_regime.index[-1]

current_start = first_date
for date in switch_dates:
    # Color de la zona anterior al cambio
    regime_was_bull = ibex_regime_clean.asof(current_start - pd.Timedelta(days=1))
    color = '#00ff00' if regime_was_bull else '#ff0000'
    ax.axvspan(current_start, date, color=color, alpha=0.03)
    
    # Línea vertical de cambio
    ax.axvline(date, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    current_start = date

# Última zona hasta el final
regime_final_bull = ibex_regime_clean.iloc[-1]
ax.axvspan(current_start, last_date, color='#00ff00' if regime_final_bull else '#ff0000', alpha=0.03)

# Líneas principales de las estrategias
ax.plot(balance_regime.index, balance_regime.values, color='#00ffcc', linewidth=3.0, label='Propuesta Final: Regime Switching (ROI +69%)')
ax.plot(balance_trailing.index, balance_trailing.values, color='#3399ff', linewidth=2.0, alpha=0.6, label='Exp 1: Trailing Stop Puro')
ax.plot(balance_original.index, balance_original.values, color='#ff3333', linewidth=2.0, alpha=0.6, label='Base TFM: Límite 48h')
ax.plot(curva_ibex.index, curva_ibex.values, color='#ffffff', linewidth=1.5, linestyle='--', alpha=0.3, label='Benchmark IBEX 35')

# Formato
formatter = ticker.StrMethodFormatter('{x:,.0f} €')
ax.yaxis.set_major_formatter(formatter)

ax.set_title('EVOLUCIÓN DE LA INVESTIGACIÓN: COMPARATIVA DE TODAS LAS ESTRATEGIAS', fontsize=18, pad=25, fontweight='bold')
ax.set_ylabel('Valor de la Cartera (EUR)', fontsize=13)
ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)

# Formato de fechas
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=0, fontsize=11)
plt.yticks(fontsize=11)

# Anotaciones finales
final_reg = balance_regime.iloc[-1]
final_tra = balance_trailing.iloc[-1]
final_ori = balance_original.iloc[-1]
ax.text(balance_regime.index[-1], final_reg, f'  {final_reg:,.0f}€', color='#00ffcc', fontweight='bold', fontsize=12, va='center')
ax.text(balance_trailing.index[-1], final_tra, f'  {final_tra:,.0f}€', color='#3399ff', fontweight='bold', fontsize=11, va='center')
ax.text(balance_original.index[-1], final_ori, f'  {final_ori:,.0f}€', color='#ff3333', fontweight='bold', fontsize=11, va='center')

plt.tight_layout()
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Comparativa TOTAL generada en: {out_img}")
