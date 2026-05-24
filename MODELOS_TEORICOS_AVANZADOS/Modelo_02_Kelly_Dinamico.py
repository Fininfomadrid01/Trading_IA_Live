"""
MODELO TEÓRICO 2: GESTIÓN DE CAPITAL KELLY-FRACTIONAL DINÁMICO
=============================================================
Teoría: Ajustar el tamaño de la posición (Sizing) proporcionalmente
a la probabilidad de éxito entregada por la IA.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT.parent / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"

CAPITAL_INICIAL = 100000.0
PROB_UMBRAL     = 0.93
SL_PCT          = 0.047
HORIZONTE       = 48

def run_experiment():
    print("Ejecutando Modelo 2: Kelly-Fractional Dinámico...")
    data = np.load(str(CACHE_PATH))
    tickers = [k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")]
    
    trades = []
    for tk in tickers:
        p = data[f"preds_{tk}"]
        idx = pd.to_datetime(data[f"idx_{tk}"])
        c = data[f"c_{tk}"]
        l = data[f"l_{tk}"]
        
        for i in range(60, len(p)):
            prob = p[i][1]
            if prob >= PROB_UMBRAL:
                # Kelly simplificado: f = (prob * win_size - loss_prob * loss_size) / win_size
                # Aquí usamos una versión lineal: alloc = base * (prob - umbral) / (1 - umbral)
                sizing_factor = (prob - PROB_UMBRAL) / (1 - PROB_UMBRAL)
                alloc = 0.05 + (0.15 * sizing_factor) # Entre 5% y 20%
                
                entry_p = c[i]
                sl = entry_p * (1 - SL_PCT)
                end = min(i + HORIZONTE, len(c)-1)
                
                exit_p = c[end]
                for j in range(i+1, end+1):
                    if l[j] <= sl:
                        exit_p = sl
                        break
                
                gain_pct = (exit_p - entry_p) / entry_p
                trades.append({"ts": idx[i], "gain": gain_pct, "alloc": alloc})
    
    df = pd.DataFrame(trades).sort_values("ts")
    
    # LIMITAR EL NÚMERO DE TRADES SIMULTÁNEOS PARA EVITAR ROI INFINITO
    # Y simular una cartera real
    capital = CAPITAL_INICIAL
    max_posiciones = 10
    posiciones_activas = []
    
    for _, row in df.iterrows():
        # Limpiar posiciones cerradas
        posiciones_activas = [p for p in posiciones_activas if p["exit_ts"] > row["ts"]]
        
        if len(posiciones_activas) < max_posiciones:
            pos_eur = (capital / max_posiciones) * (row["alloc"] / 0.20) # Normalizado
            pnl = pos_eur * row["gain"]
            capital += pnl
            posiciones_activas.append({"exit_ts": row["ts"] + pd.Timedelta(hours=HORIZONTE)})
        
    roi = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    print(f"  Capital Final: {capital:,.2f} EUR")
    print(f"  ROI Final: {roi:+.2f}%")

if __name__ == "__main__":
    run_experiment()
