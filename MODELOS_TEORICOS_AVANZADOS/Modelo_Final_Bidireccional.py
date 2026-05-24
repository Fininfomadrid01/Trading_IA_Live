"""
SISTEMA HÍBRIDO BIDIRECCIONAL: LONG + SHORT
===========================================
Incorpora operativa en corto (Short) al modelo híbrido:
1. Filtro: Compresión de volatilidad (Modelo 05).
2. Dirección: Determinado por las clases ALC_CONT (Long) y BAJ_CONT (Short).
3. Gestión: Kelly Dinámico bidireccional.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT.parent / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"

CAPITAL_INICIAL = 100000.0
PROB_UMBRAL     = 0.93
SL_PCT          = 0.05
HORIZONTE       = 48
MAX_POSICIONES  = 10

def run_bidirectional_system():
    t0 = time.time()
    print("\nEJECUTANDO SISTEMA HÍBRIDO BIDIRECCIONAL (LONG + SHORT)...")
    print("=" * 60)
    
    data = np.load(str(CACHE_PATH))
    tickers = [k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")]
    
    all_potential_trades = []
    
    print(f"  Procesando {len(tickers)} activos...")
    for tk in tickers:
        p = data[f"preds_{tk}"]
        c = data[f"c_{tk}"]
        h = data[f"h_{tk}"]
        l = data[f"l_{tk}"]
        idx = pd.to_datetime(data[f"idx_{tk}"])
        
        # Filtro de Volatilidad
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
        vol = pd.Series(tr).rolling(20).mean()
        vol_relative = vol / pd.Series(c).rolling(20).mean()
        vol_quantile = vol_relative.rolling(100).quantile(0.3)
        
        for i in range(100, len(p)):
            prob_long  = p[i][1] # ALC_CONT
            prob_short = p[i][2] # BAJ_CONT
            
            # Solo entrar si hay compresión de volatilidad
            if vol_relative[i] < vol_quantile[i]:
                direction = None
                prob = 0
                
                if prob_long >= PROB_UMBRAL:
                    direction = "LONG"
                    prob = prob_long
                elif prob_short >= PROB_UMBRAL:
                    direction = "SHORT"
                    prob = prob_short
                
                if direction:
                    sizing_factor = (prob - PROB_UMBRAL) / (1 - PROB_UMBRAL)
                    alloc = 0.05 + (0.15 * sizing_factor)
                    
                    entry_p = c[i]
                    end = min(i + HORIZONTE, len(c)-1)
                    exit_p = c[end]
                    
                    if direction == "LONG":
                        sl = entry_p * (1 - SL_PCT)
                        for j in range(i+1, end+1):
                            if l[j] <= sl:
                                exit_p = sl
                                break
                        gain = (exit_p - entry_p) / entry_p
                    else: # SHORT
                        sl = entry_p * (1 + SL_PCT)
                        for j in range(i+1, end+1):
                            if h[j] >= sl:
                                exit_p = sl
                                break
                        gain = (entry_p - exit_p) / entry_p # Invertido para Short
                    
                    all_potential_trades.append({
                        "ts": idx[i],
                        "exit_ts": idx[end],
                        "gain": gain,
                        "alloc": alloc,
                        "tk": tk,
                        "dir": direction
                    })
    
    # Simulación de Cartera
    df_trades = pd.DataFrame(all_potential_trades).sort_values("ts")
    capital = CAPITAL_INICIAL
    posiciones_activas = {}
    
    print(f"  Simulando cartera bidireccional sobre {len(df_trades)} señales...")
    for _, trade in df_trades.iterrows():
        posiciones_activas = {tk: ets for tk, ets in posiciones_activas.items() if ets > trade["ts"]}
        
        if trade["tk"] not in posiciones_activas and len(posiciones_activas) < MAX_POSICIONES:
            pos_eur = (capital / MAX_POSICIONES) * (trade["alloc"] / 0.15)
            pos_eur = min(pos_eur, capital * 0.20)
            
            capital += pos_eur * trade["gain"]
            posiciones_activas[trade["tk"]] = trade["exit_ts"]

    roi = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    print("\nRESULTADOS DEL SISTEMA BIDIRECCIONAL (LONG+SHORT):")
    print("-" * 45)
    print(f"  Capital Final:    {capital:,.2f} EUR")
    print(f"  ROI Total:        {roi:+.2f}%")
    print(f"  Total Trades:     {len(df_trades)}")
    print(f"  Longs:            {len(df_trades[df_trades['dir']=='LONG'])}")
    print(f"  Shorts:           {len(df_trades[df_trades['dir']=='SHORT'])}")
    print(f"  Tiempo ejecución: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run_bidirectional_system()
