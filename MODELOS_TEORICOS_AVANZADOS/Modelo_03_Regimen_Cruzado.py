"""
MODELO TEÓRICO 3: INFERENCIA DE RÉGIMEN CRUZADO (MARKET OVERLAY)
===============================================================
Teoría: Solo operar activos locales (España) si los activos globales
líderes (QQQ/NASDAQ) están en régimen alcista.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT.parent / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"

def run_experiment():
    print("Ejecutando Modelo 3: Régimen Cruzado (Filtro NASDAQ)...")
    data = np.load(str(CACHE_PATH))
    
    # 1. Definir régimen global (NASDAQ - QQQ)
    q_idx = pd.to_datetime(data["idx_QQQ"])
    q_c = data["c_QQQ"]
    df_q = pd.DataFrame({"close": q_c}, index=q_idx)
    # Usar una media muy corta (SMA 20) para detectar cambios rápidos
    sma_filter = df_q["close"].rolling(20).mean()
    regime_global = df_q["close"] > sma_filter
    
    # Ampliar universo ES
    tickers_es = [tk for tk in data.keys() if tk.startswith("preds_") and tk.replace("preds_","") not in ['QQQ','AAPL','MSFT','NVDA','TSLA','BTC','ETH','GOLD','OIL','SHY','FTSE']]
    tickers_es = [tk.replace("preds_","") for tk in tickers_es]
    
    trades = []
    for tk in tickers_es:
        if f"preds_{tk}" not in data: continue
        p = data[f"preds_{tk}"]
        idx = pd.to_datetime(data[f"idx_{tk}"])
        c = data[f"c_{tk}"]
        l = data[f"l_{tk}"]
        
        # Sincronizar régimen global con el índice del ticker actual
        regime_tk = regime_global.reindex(idx, method='ffill').fillna(False)
        
        for i in range(200, len(p)):
            if regime_tk.iloc[i] and p[i][1] >= 0.93:
                entry_p = c[i]
                sl = entry_p * 0.95
                end = min(i+48, len(c)-1)
                exit_p = c[end]
                for j in range(i+1, end+1):
                    if l[j] <= sl: exit_p = sl; break
                trades.append((exit_p - entry_p)/entry_p)
                
    if trades:
        print(f"  Trades con filtro global: {len(trades)}")
        print(f"  Ganancia Media: {np.mean(trades)*100:+.3f}%")
    else:
        print("  No se generaron trades con el filtro.")

if __name__ == "__main__":
    run_experiment()
