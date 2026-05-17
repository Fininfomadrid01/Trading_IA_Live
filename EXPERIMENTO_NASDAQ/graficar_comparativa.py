import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- DATOS NORMALIZADOS (RENTABILIDAD ANUALIZADA - CAGR) ---
# IBEX: (1 + 0.688)^(1/7.4) - 1  (Periodo 2018 - Mayo 2025 aprox)
# NASDAQ: (1 + 0.6015)^(1/2.9) - 1
# S&P 500: (1 + 0.3356)^(1/2.9) - 1
# FTSE 100: (1 + 0.4661)^(1/2.8) - 1

mercados = ['IBEX 35', 'NASDAQ 100', 'S&P 500', 'FTSE 100']
roi_total = [68.80, 60.15, 33.56, 46.61]
periodos = [7.4, 2.9, 2.9, 2.8] # Años

# Cálculo exacto del CAGR (%)
roi_anualizado = [((1 + r/100)**(1/p) - 1) * 100 for r, p in zip(roi_total, periodos)]
profit_factor = [1.20, 1.20, 1.23, 1.20] 

# Configuración de estilo
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(14, 8))

# BARRAS DE ROI ANUALIZADO
colors = ['#00ffcc', '#00e6b8', '#00cca3', '#00b38f']
bars = ax1.bar(mercados, roi_anualizado, color=colors, alpha=0.8, width=0.6, label='ROI Anualizado (%)')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', 
             color='white', fontweight='bold', fontsize=12)

ax1.set_ylabel('Rentabilidad Anualizada (CAGR %)', color='#00ffcc', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 25)
ax1.tick_params(axis='y', labelcolor='#00ffcc')
ax1.grid(axis='y', linestyle='--', alpha=0.2)

# LÍNEA DE PROFIT FACTOR
ax2 = ax1.twinx()
ax2.plot(mercados, profit_factor, color='#ff3366', marker='o', linewidth=4, markersize=12, label='Profit Factor')
ax2.set_ylabel('Profit Factor (Calidad)', color='#ff3366', fontsize=12, fontweight='bold')
ax2.set_ylim(1.10, 1.30)
ax2.tick_params(axis='y', labelcolor='#ff3366')

for i, txt in enumerate(profit_factor):
    ax2.annotate(f'PF: {txt:.2f}', (mercados[i], profit_factor[i]), 
                 textcoords="offset points", xytext=(0,15), ha='center', 
                 color='#ff3366', fontweight='bold', fontsize=11)

plt.title('COMPARATIVA DE RENDIMIENTO ANUALIZADO (Normalizado)', 
          fontsize=18, fontweight='bold', pad=25, color='white')

info_text = "Nota: Rentabilidades normalizadas anualmente (CAGR) para permitir una comparación justa entre periodos distintos."
plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=11, bbox={"facecolor":"grey", "alpha":0.1, "pad":8})

plt.tight_layout()

BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
SAVE_PATH = BASE_DIR / "Documentacion_TFM" / "COMPARATIVA_GENERALIZACION_GLOBAL.png"
plt.savefig(SAVE_PATH, dpi=150)
print(f"Gráfico anualizado guardado: {SAVE_PATH}")
