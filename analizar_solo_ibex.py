import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"
ISINS_IBEX = ROOT / "isins_ibex35.csv"
TXT_IBEX = BASE_DIR / "HISTORICOS" / "Historicos IBEX 35 19052026.txt"

CAPITAL = 100000.0
PROB_UMBRAL = 0.93
SL_PCT = 0.047
HORIZONTE = 48

def load_ibex_prices(txt_path, isin_path):
    df_isin = pd.read_csv(isin_path)
    isin_col = 'CVALISO' if 'CVALISO' in df_isin.columns else 'ISIN'
    isin_to_ticker = dict(zip(df_isin[isin_col].str.strip(), df_isin['VALOR'].str.strip()))
    
    ticker_data = {}
    with open(txt_path, 'r', encoding='latin1') as f:
        for line in f:
            if not line.startswith("0001"): continue
            parts = line.split(",")
            isin = parts[1].strip().split()[0]
            if isin[:2] in ("XX","YY","ZZ"): isin = isin[2:]
            ticker = isin_to_ticker.get(isin)
            if not ticker: continue
            try:
                if ticker not in ticker_data: ticker_data[ticker] = []
                ticker_data[ticker].append({
                    'date': pd.to_datetime(parts[2].strip(), format='%Y%m%d'),
                    'low': float(parts[5]), 'close': float(parts[6])
                })
            except: continue
    return {tk: pd.DataFrame(rows).set_index('date') for tk, rows in ticker_data.items()}

def run_ibex_only():
    print("Analizando SOLO valores del IBEX 35...")
    prices = load_ibex_prices(TXT_IBEX, ISINS_IBEX)
    data = np.load(str(CACHE_PATH))
    
    trades = []
    for tk in prices.keys():
        if f"preds_{tk}" not in data: continue
        p = data[f"preds_{tk}"]
        idx = pd.to_datetime(data[f"idx_{tk}"])
        c = prices[tk].reindex(idx, method='ffill').bfill()['close'].values
        l = prices[tk].reindex(idx, method='ffill').bfill()['low'].values
        
        last_exit = -1
        for i in range(60, len(p)):
            if i <= last_exit: continue
            if p[i][1] >= PROB_UMBRAL:
                entry_p = c[i]
                sl = entry_p * (1 - SL_PCT)
                exit_i = min(i + HORIZONTE, len(c) - 1)
                res = "TIME"
                for j in range(i+1, exit_i+1):
                    if l[j] <= sl: exit_i = j; res = "SL"; break
                exit_p = sl if res == "SL" else c[exit_i]
                trades.append((exit_p - entry_p) / entry_p)
                last_exit = exit_i + 3
                
    if not trades: return 0, 0
    avg_gain = np.mean(trades)
    wr = len([t for t in trades if t > 0]) / len(trades)
    return avg_gain, wr, len(trades)

avg, wr, count = run_ibex_only()
print(f"\nRESULTADOS SOLO IBEX 35:")
print(f"  Trades: {count}")
print(f"  Win Rate: {wr*100:.2f}%")
print(f"  Ganancia Media: {avg*100:+.3f}%")
