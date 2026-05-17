import pandas as pd
import numpy as np

# Cargar trades
df = pd.read_csv("Resultados_Regime.csv")
df['entry_ts'] = pd.to_datetime(df['entry_ts'])
df['exit_ts'] = pd.to_datetime(df['exit_ts'])

# Filtrar por periodo 2025-2026
df_filtered = df[(df['entry_ts'] >= '2025-01-01') & (df['entry_ts'] <= '2026-12-31')].copy()

# Calcular Volumen (Nº de Acciones / Unidades)
df_filtered['Volumen_Unidades'] = (df_filtered['pos_eur'] / df_filtered['entry_price']).round(0).astype(int)

# Reordenar y renombrar columnas para que sean muy legibles para el tribunal
cols_to_keep = {
    'ticker': 'Ticker',
    'tipo': 'Direccion',
    'regimen': 'Regimen_Mercado',
    'entry_ts': 'Fecha_Entrada',
    'exit_ts': 'Fecha_Salida',
    'entry_price': 'Precio_Entrada',
    'exit_price': 'Precio_Salida',
    'Volumen_Unidades': 'Volumen_Acciones',
    'pos_eur': 'Exposicion_EUR',
    'ganancia_pct': 'Ganancia_Pct',
    'pnl_neto': 'PnL_Neto_EUR',
    'motivo': 'Motivo_Salida',
    'prob': 'Confianza_IA'
}

df_report = df_filtered[list(cols_to_keep.keys())].rename(columns=cols_to_keep)

# Formatear columnas
df_report['Ganancia_Pct'] = (df_report['Ganancia_Pct'] * 100).round(2).astype(str) + '%'
df_report['Confianza_IA'] = (df_report['Confianza_IA'] * 100).round(2).astype(str) + '%'
df_report['Precio_Entrada'] = df_report['Precio_Entrada'].round(4)
df_report['Precio_Salida'] = df_report['Precio_Salida'].round(4)
df_report['Exposicion_EUR'] = df_report['Exposicion_EUR'].round(2)
df_report['PnL_Neto_EUR'] = df_report['PnL_Neto_EUR'].round(2)

# Ordenar cronológicamente
df_report = df_report.sort_values(by='Fecha_Entrada')

# Guardar a CSV limpio de entrega
df_report.to_csv("Operaciones_TFM_2025_2026.csv", index=False, encoding='utf-8')

# Calcular estadísticas del periodo
wins = df_filtered[df_filtered['pnl_neto'] > 0]
losses = df_filtered[df_filtered['pnl_neto'] <= 0]
win_rate = (len(wins) / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
total_pnl = df_filtered['pnl_neto'].sum()

profit_factor = 0
if losses['pnl_neto'].sum() != 0:
    profit_factor = abs(wins['pnl_neto'].sum() / losses['pnl_neto'].sum())

print("-" * 50)
print(f"ESTADÍSTICAS OPERATIVA 2025-2026:")
print("-" * 50)
print(f"Total Operaciones:    {len(df_report)}")
print(f"Win Rate:             {win_rate:.2f}%")
print(f"PnL Neto Total:       {total_pnl:,.2f} EUR")
print(f"Profit Factor:        {profit_factor:.2f}")
print("-" * 50)

# Mostrar las primeras 15 operaciones para previsualización
print(df_report.head(15).to_string(index=False))
