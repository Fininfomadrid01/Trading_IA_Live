"""
MODELO TEÓRICO 1: ROTACIÓN DE MOMENTUM POR PROBABILIDAD IA
==========================================================
Teoría: Solo ejecutar las N mejores señales diarias por probabilidad,
siempre que superen un umbral de fuerza relativa.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT.parent / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"
META_PATH = ROOT.parent / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_meta.csv"

TOP_N_DIARIO = 5
PROB_MINIMA  = 0.95
CAPITAL      = 100000.0
SL_PCT       = 0.05
HORIZONTE    = 48

def run_experiment():
    print(f"Ejecutando Modelo 1: Top {TOP_N_DIARIO} señales diarias...")
    
    data = np.load(str(CACHE_PATH))
    meta = pd.read_csv(META_PATH)
    tickers = meta["ticker"].tolist()
    
    all_signals = []
    for tk in tickers:
        if f"preds_{tk}" not in data: continue
        preds = data[f"preds_{tk}"]
        idx = pd.to_datetime(data[f"idx_{tk}"])
        c = data[f"c_{tk}"]
        l = data[f"l_{tk}"]
        
        for i in range(60, len(preds)):
            prob = preds[i][1] # Clase ALC_CONT
            if prob >= PROB_MINIMA:
                all_signals.append({
                    "ts": idx[i], "tk": tk, "prob": prob,
                    "entry_p": c[i], "sl": c[i]*(1-SL_PCT),
                    "idx_entry": i, "c": c, "l": l
                })
    
    df_sig = pd.DataFrame(all_signals)
    if df_sig.empty:
        print("No se generaron señales.")
        return
    
    # Convertir prob a float64 para evitar el error de float16 en índices
    df_sig["prob"] = df_sig["prob"].astype(np.float64)
    df_sig = df_sig.sort_values(["ts", "prob"], ascending=[True, False])
    
    # Filtrar Top N por día
    df_sig["fecha"] = df_sig["ts"].dt.date
    df_top = df_sig.groupby("fecha").head(TOP_N_DIARIO).copy()
    
    # Simulación simple
    results = []
    for _, row in df_top.iterrows():
        c = row["c"]
        l = row["l"]
        start = row["idx_entry"]
        end = min(start + HORIZONTE, len(c)-1)
        
        exit_p = c[end]
        for j in range(start+1, end+1):
            if l[j] <= row["sl"]:
                exit_p = row["sl"]
                break
        
        results.append((exit_p - row["entry_p"]) / row["entry_p"])
    
    print(f"  Trades ejecutados: {len(results)}")
    print(f"  Win Rate: {len([r for r in results if r > 0])/len(results)*100:.2f}%")
    print(f"  Ganancia Media: {np.mean(results)*100:+.3f}%")

if __name__ == "__main__":
    run_experiment()
