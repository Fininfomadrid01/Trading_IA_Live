"""
MODELO TEÓRICO 5: ARBITRAJE DE VOLATILIDAD PREDICTIVA
=====================================================
Teoría: Solo entrar si la IA predice subida Y el activo está en
un periodo de baja volatilidad (compresión), esperando la expansión.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT.parent / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"

def run_experiment():
    print("Ejecutando Modelo 5: Arbitraje de Volatilidad Predictiva...")
    data = np.load(str(CACHE_PATH))
    tickers = [k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")]
    
    trades = []
    for tk in tickers:
        p = data[f"preds_{tk}"]
        c = data[f"c_{tk}"]
        h = data[f"h_{tk}"]
        l = data[f"l_{tk}"]
        
        # Calcular volatilidad (ATR simplificado de 20 periodos)
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
        vol = pd.Series(tr).rolling(20).mean()
        vol_relative = vol / pd.Series(c).rolling(20).mean()
        
        # Pre-calcular el cuantil para evitar el rolling lento dentro del bucle
        vol_quantile = vol_relative.rolling(100).quantile(0.3)
        
        for i in range(60, len(p)):
            # FILTRO: Probabilidad alta + Volatilidad relativa baja (percentil 30)
            if p[i][1] >= 0.93 and vol_relative[i] < vol_quantile[i]:
                entry_p = c[i]
                sl = entry_p * (1 - 0.05)
                end = min(i+48, len(c)-1)
                exit_p = c[end]
                for j in range(i+1, end+1):
                    if l[j] <= sl: exit_p = sl; break
                trades.append((exit_p - entry_p)/entry_p)

    if trades:
        print(f"  Trades con filtro de volatilidad: {len(trades)}")
        print(f"  Ganancia Media: {np.mean(trades)*100:+.3f}%")
        print(f"  Win Rate: {len([t for t in trades if t > 0])/len(trades)*100:.2f}%")

if __name__ == "__main__":
    run_experiment()
