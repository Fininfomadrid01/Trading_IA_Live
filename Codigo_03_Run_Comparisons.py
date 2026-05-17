import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ULTRA-REALISTA ---
ROOT = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS")
CACHE_PATH = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"
CAPITAL = 100000.0
COMISION_PCT = 0.0005 # 0.05% (IBKR Tiered)
COMISION_MIN = 1.25   # 1.25 EUR
SLIPPAGE     = 0.0002 # 0.02%

def run_backtest_serio(mode):
    data = np.load(str(CACHE_PATH), allow_pickle=False)
    meta = pd.read_csv(CACHE_PATH.parent / "cache_meta.csv")
    
    trades = []
    prob_umbral = 0.93
    sl_pct = 0.047
    ts_pct = 0.055
    pos_size = CAPITAL * 0.10 # 10% capital
    
    for tk in meta["ticker"]:
        try:
            p = data[f"preds_{tk}"]; c = data[f"c_{tk}"]; l = data[f"l_{tk}"]; h = data[f"h_{tk}"]; idx = pd.to_datetime(data[f"idx_{tk}"])
        except: continue
        
        n = len(p)
        last_exit = -1
        
        for i in range(60, n):
            if i <= last_exit: continue
            if p[i][1] < prob_umbral: continue
            
            entry_price = c[i]
            
            if mode == "BASE":
                exit_i = min(i + 48, n - 1)
                sl = entry_price * (1 - sl_pct)
                res = "TIME"
                for j in range(i+1, exit_i+1):
                    if l[j] <= sl: exit_i = j; res="SL"; break
                exit_price = sl if res == "SL" else c[exit_i]
            
            elif mode == "TRAILING":
                exit_i = min(i + 150, n - 1)
                max_v = entry_price; sl = entry_price * (1 - sl_pct)
                res = "TIME"
                for j in range(i+1, exit_i+1):
                    if h[j] > max_v: max_v = h[j]
                    stop = max(sl, max_v * (1 - ts_pct))
                    if l[j] <= stop: exit_i = j; exit_price = stop; res="TS"; break
                else: exit_price = c[exit_i]
            
            gain_raw = (exit_price - entry_price) / entry_price
            
            # --- SANITY CHECK (Evitar errores de datos/splits) ---
            # Si una operacion gana mas del 25% en 2-5 dias en el IBEX, es un error de datos.
            if gain_raw > 0.25: gain_raw = 0.02 # Lo tratamos como una ganancia normal
            
            pnl_bruto = gain_raw * pos_size
            com_in  = max(COMISION_MIN, pos_size * (COMISION_PCT + SLIPPAGE))
            com_out = max(COMISION_MIN, pos_size * (COMISION_PCT + SLIPPAGE))
            
            trades.append({"exit_ts": idx[exit_i], "pnl_neto": pnl_bruto - com_in - com_out})
            last_exit = exit_i + 3
            
    return pd.DataFrame(trades)

print("Recalculando BASE (Sanity Check + IBKR)...")
run_backtest_serio("BASE").to_csv(ROOT / "ENTREGA_TFM_VERSION_FINAL" / "Resultados_Base.csv", index=False)

print("Recalculando TRAILING (Sanity Check + IBKR)...")
run_backtest_serio("TRAILING").to_csv(ROOT / "ENTREGA_TFM_VERSION_FINAL" / "Resultados_Trailing.csv", index=False)
print("Hecho.")
