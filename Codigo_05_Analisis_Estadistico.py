import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Rutas de archivos
files = {
    "Base TFM (48h Fijo)": r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_MM_Avanzada.csv",
    "Exp 1: Trailing Puro": r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_Trailing.csv",
    "Propuesta Final: Regime": r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Resultados_Regime.csv"
}

stats = []

for name, path in files.items():
    df = pd.read_csv(path)
    
    total_ops = len(df)
    ganadoras = df[df['pnl_neto'] > 0]
    perdedoras = df[df['pnl_neto'] <= 0]
    
    wr = len(ganadoras) / total_ops * 100
    avg_w = ganadoras['ganancia_pct'].mean() * 100
    avg_l = perdedoras['ganancia_pct'].mean() * 100
    
    profit_bruto = ganadoras['pnl_neto'].sum()
    perdida_bruta = abs(perdedoras['pnl_neto'].sum())
    pf = profit_bruto / perdida_bruta if perdida_bruta > 0 else np.inf
    
    stats.append({
        "Estrategia": name,
        "Operaciones": total_ops,
        "Win Rate (%)": round(wr, 2),
        "Avg Win (%)": round(avg_w, 2),
        "Avg Loss (%)": round(avg_l, 2),
        "Profit Factor": round(pf, 2)
    })

df_stats = pd.DataFrame(stats)
print("\nRESUMEN ESTADÍSTICO COMPARATIVO:")
print(df_stats.to_string(index=False))

# Guardar informe en CSV
df_stats.to_csv(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\resultados\Analisis_Estadistico_Sistemas.csv", index=False)

# Generar Gráfico de Barras
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Número de Operaciones
ax1.bar(df_stats['Estrategia'], df_stats['Operaciones'], color=['#ff3333', '#3399ff', '#00ffcc'], alpha=0.8)
ax1.set_title('Volumen de Operaciones', fontsize=14)
ax1.set_ylabel('Nº de Trades')
for i, v in enumerate(df_stats['Operaciones']):
    ax1.text(i, v + 20, str(v), ha='center', fontweight='bold')

# Gráfico 2: Win Rate
ax2.bar(df_stats['Estrategia'], df_stats['Win Rate (%)'], color=['#ff3333', '#3399ff', '#00ffcc'], alpha=0.8)
ax2.set_title('Tasa de Acierto (Win Rate)', fontsize=14)
ax2.set_ylabel('% de Acierto')
ax2.set_ylim(0, 60)
for i, v in enumerate(df_stats['Win Rate (%)']):
    ax2.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\Analisis_Estadistico_Barras.png", dpi=300)
print(f"\nGráfico guardado en: Analisis_Estadistico_Barras.png")
