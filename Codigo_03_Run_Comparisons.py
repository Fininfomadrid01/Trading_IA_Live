import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ULTRA-REALISTA ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"
TXT_PATHS = [
    BASE_DIR / "HISTORICOS" / "Historicos IBEX 35 19052026.txt",
    BASE_DIR / "HISTORICOS" / "Historicos mercado continuo 19052026.txt"
]
ISIN_PATHS = [ROOT / "isins_ibex35.csv", ROOT / "isins_mercadocontinuo.csv"]
CAPITAL = 100000.0
COMISION_PCT = 0.0005 # 0.05% (IBKR Tiered)
COMISION_MIN = 1.25   # 1.25 EUR
SLIPPAGE     = 0.0002 # 0.02%

def load_txt_prices(txt_paths, isin_paths):
    isin_to_ticker = {}
    for isin_path in isin_paths:
        df_isin = pd.read_csv(isin_path)
        # Handle different column names (CVALISO vs ISIN)
        isin_col = 'CVALISO' if 'CVALISO' in df_isin.columns else 'ISIN'
        df_isin = df_isin.rename(columns={isin_col: 'isin'})
        df_isin['isin'] = df_isin['isin'].astype(str).str.strip()
        df_isin['VALOR'] = df_isin['VALOR'].astype(str).str.strip()
        isin_to_ticker.update(dict(zip(df_isin['isin'], df_isin['VALOR'])))
    
    ticker_data = {}
    for txt_path in txt_paths:
        if not txt_path.exists(): continue
        with open(txt_path, 'r', encoding='latin1') as f:
            for line in f:
                if not line.startswith("0001"): continue
                parts = line.split(",")
                if len(parts) < 12: continue
                raw_code = parts[1].strip()
                subparts = [p for p in raw_code.split() if p]
                if not subparts: continue
                isin = subparts[0]
                if isin.startswith("XX") or isin.startswith("YY") or isin.startswith("ZZ"):
                    isin = isin[2:]
                
                ticker = isin_to_ticker.get(isin)
                if not ticker:
                    if isin == "NL0015001FS8": ticker = "FER"
                    elif isin == "LU1598757687": ticker = "MTS"
                    elif isin == "ES0113307062": ticker = "BKIA"
                    else: continue
                    
                try:
                    date = parts[2].strip()
                    h = float(parts[4].strip())
                    l = float(parts[5].strip())
                    c = float(parts[6].strip())
                    v = float(parts[11].strip())
                    
                    if ticker not in ticker_data:
                        ticker_data[ticker] = []
                    ticker_data[ticker].append({
                        'date': date, 'high': h, 'low': l, 'close': c, 'volume': v
                    })
                except: continue
                
    cache_prices = {}
    for ticker, rows in ticker_data.items():
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date').drop_duplicates('date').set_index('date')
        cache_prices[ticker] = df
        
    return cache_prices

def run_backtest_serio(mode, txt_prices):
    data = np.load(str(CACHE_PATH), allow_pickle=False)
    meta = pd.read_csv(CACHE_PATH.parent / "cache_meta.csv")
    
    trades = []
    prob_umbral = 0.93
    sl_pct = 0.047
    ts_pct = 0.055
    pos_size = CAPITAL * 0.10 # 10% capital
    
    for tk in meta["ticker"]:
        try:
            idx = pd.to_datetime(data[f"idx_{tk}"])
            p   = data[f"preds_{tk}"]
            if tk in txt_prices:
                df_r = txt_prices[tk].reindex(idx, method='ffill').bfill()
                c = df_r['close'].values.astype(np.float32)
                l = df_r['low'].values.astype(np.float32)
                h = df_r['high'].values.astype(np.float32)
            else:
                c = data[f"c_{tk}"]
                l = data[f"l_{tk}"]
                h = data[f"h_{tk}"]
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
            
            pnl_bruto = gain_raw * pos_size
            com_in  = max(COMISION_MIN, pos_size * (COMISION_PCT + SLIPPAGE))
            com_out = max(COMISION_MIN, pos_size * (COMISION_PCT + SLIPPAGE))
            
            trades.append({"exit_ts": idx[exit_i], "pnl_neto": pnl_bruto - com_in - com_out})
            last_exit = exit_i + 3
            
    return pd.DataFrame(trades)

print("Cargando precios locales de España (Historicos IBEX + Continuo)...", end=" ", flush=True)
txt_prices = load_txt_prices(TXT_PATHS, ISIN_PATHS)
print(f"OK ({len(txt_prices)} tickers)")

print("Recalculando BASE (Sanity Check + IBKR)...")
run_backtest_serio("BASE", txt_prices).to_csv(BASE_DIR / "Resultados_Base.csv", index=False)

print("Recalculando TRAILING (Sanity Check + IBKR)...")
run_backtest_serio("TRAILING", txt_prices).to_csv(BASE_DIR / "Resultados_Trailing.csv", index=False)
print("Hecho.")
