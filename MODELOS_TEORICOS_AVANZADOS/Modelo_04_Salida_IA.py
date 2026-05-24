"""
MODELO TEÓRICO 4: SALIDA POR DECAIMIENTO DE ATENCIÓN (IA EXIT)
==============================================================
Teoría: En lugar de salir por tiempo fijo, salimos cuando la 
probabilidad de la IA para la clase alcista baja de un umbral.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT.parent / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"

UMBRAL_ENTRADA = 0.95
UMBRAL_SALIDA  = 0.75 # Salimos si la confianza baja del 75%

def run_experiment():
    print("Ejecutando Modelo 4: Salida por Decaimiento de Atención...")
    data = np.load(str(CACHE_PATH))
    tickers = ['QQQ', 'NVDA', 'BTC', 'ETH'] # Activos con alta volatilidad
    
    all_gains = []
    for tk in tickers:
        p = data[f"preds_{tk}"]
        c = data[f"c_{tk}"]
        l = data[f"l_{tk}"]
        
        in_pos = False
        entry_p = 0
        
        for i in range(60, len(p)):
            prob = p[i][1]
            
            if not in_pos and prob >= UMBRAL_ENTRADA:
                in_pos = True
                entry_p = c[i]
                sl = entry_p * 0.95
            
            elif in_pos:
                # Condición de salida: Decaimiento de Probabilidad o SL
                if l[i] <= sl:
                    all_gains.append((sl - entry_p)/entry_p)
                    in_pos = False
                elif prob < UMBRAL_SALIDA:
                    all_gains.append((c[i] - entry_p)/entry_p)
                    in_pos = False
                elif i == len(p) - 1:
                    all_gains.append((c[i] - entry_p)/entry_p)
                    in_pos = False

    if all_gains:
        print(f"  Trades con salida IA: {len(all_gains)}")
        print(f"  Ganancia Media: {np.mean(all_gains)*100:+.3f}%")
        print(f"  Win Rate: {len([g for g in all_gains if g > 0])/len(all_gains)*100:.2f}%")

if __name__ == "__main__":
    run_experiment()
