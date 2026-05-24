"""
SISTEMA HÍBRIDO AVANZADO: VOLATILIDAD PREDICTIVA + KELLY DINÁMICO
================================================================
Este modelo combina las dos mejores teorías detectadas:
1. Filtro de Entrada: Solo entra en señales de alta probabilidad IA 
   cuando hay compresión de volatilidad (Modelo 05).
2. Gestión de Capital: Ajusta el tamaño de la posición según la 
   confianza de la IA (Kelly Dinámico - Modelo 02).
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

def run_hybrid_system():
    t0 = time.time()
    print("\nEJECUTANDO SISTEMA HÍBRIDO (VOLATILIDAD + KELLY)...")
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
        
        # 1. FILTRO DE VOLATILIDAD (Modelo 05)
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
        vol = pd.Series(tr).rolling(20).mean()
        vol_relative = vol / pd.Series(c).rolling(20).mean()
        vol_quantile = vol_relative.rolling(100).quantile(0.3) # Compresión (percentil 30)
        
        for i in range(100, len(p)):
            prob = p[i][1]
            # Condición Híbrida: Alta Probabilidad + Baja Volatilidad
            if prob >= PROB_UMBRAL and vol_relative[i] < vol_quantile[i]:
                # 2. SIZING KELLY DINÁMICO (Modelo 02)
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
                
                all_potential_trades.append({
                    "ts": idx[i],
                    "exit_ts": idx[end],
                    "gain": (exit_p - entry_p) / entry_p,
                    "alloc": alloc,
                    "tk": tk
                })
    
    # 3. SIMULACIÓN DE CARTERA
    df_trades = pd.DataFrame(all_potential_trades).sort_values("ts")
    
    # FILTRO DE SEÑALES SIMULTÁNEAS: Solo una señal por ticker al mismo tiempo
    # y máximo 10 señales totales en el mercado para evitar sobre-operar
    capital = CAPITAL_INICIAL
    posiciones_activas = {} # tk -> exit_ts
    historial_equity = []
    
    print(f"  Simulando cartera sobre {len(df_trades)} señales detectadas...")
    for _, trade in df_trades.iterrows():
        # Cerrar posiciones que ya terminaron
        posiciones_activas = {tk: ets for tk, ets in posiciones_activas.items() if ets > trade["ts"]}
        
        if trade["tk"] not in posiciones_activas and len(posiciones_activas) < MAX_POSICIONES:
            # Asignación de capital real (Kelly Dinámico)
            # El alloc de Kelly se usa para determinar qué parte del capital disponible arriesgar
            pos_eur = (capital / MAX_POSICIONES) * (trade["alloc"] / 0.15) # Normalización
            
            # Limitar posición para no descapitalizar
            pos_eur = min(pos_eur, capital * 0.20) 
            
            pnl = pos_eur * trade["gain"]
            capital += pnl
            posiciones_activas[trade["tk"]] = trade["exit_ts"]
            historial_equity.append({"ts": trade["ts"], "equity": capital})

    roi = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    print("\nRESULTADOS DEL SISTEMA HÍBRIDO:")
    print("-" * 30)
    print(f"  Capital Final:    {capital:,.2f} EUR")
    print(f"  ROI Total:        {roi:+.2f}%")
    print(f"  Total Trades:     {len(df_trades)}")
    print(f"  Tiempo ejecución: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run_hybrid_system()
