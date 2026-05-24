import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS")
cache_path = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"
txt_path = ROOT / "ENTREGA_TFM_VERSION_FINAL" / "HISTORICOS" / "Historicos IBEX 35 19052026.txt"
isins_path = ROOT / "isins_ibex35.csv"

def load_adjusted_txt_prices(txt_path, isins_file):
    # Load ISIN mapping
    ibex_isins = pd.read_csv(isins_file)
    ibex_isins = ibex_isins.rename(columns={'CVALISO': 'isin'})
    ibex_isins['isin'] = ibex_isins['isin'].astype(str).str.strip()
    ibex_isins['VALOR'] = ibex_isins['VALOR'].astype(str).str.strip()
    isin_to_ticker = dict(zip(ibex_isins['isin'], ibex_isins['VALOR']))
    
    # Read the file
    ticker_data = {}
    with open(txt_path, 'r', encoding='latin1') as f:
        for line in f:
            if not line.startswith("0001"):
                continue
            parts = line.split(",")
            if len(parts) < 12:
                continue
            raw_code = parts[1].strip()
            subparts = [p for p in raw_code.split() if p]
            if not subparts:
                continue
            isin = subparts[0]
            if isin.startswith("XX") or isin.startswith("YY") or isin.startswith("ZZ"):
                isin = isin[2:]
            
            ticker = isin_to_ticker.get(isin)
            if not ticker:
                if isin == "NL0015001FS8": ticker = "FER"
                elif isin == "LU1598757687": ticker = "MTS"
                elif isin == "ES0113307062": ticker = "BKIA"
                else: continue
                
            date = parts[2].strip()
            o = float(parts[3].strip())
            h = float(parts[4].strip())
            l = float(parts[5].strip())
            c = float(parts[6].strip())
            adj = float(parts[9].strip())
            v = float(parts[11].strip())
            
            if ticker not in ticker_data:
                ticker_data[ticker] = []
            ticker_data[ticker].append({
                'date': date, 'open': o, 'high': h, 'low': l, 'close': c, 'adj_factor': adj, 'volume': v
            })
            
    # Adjust each ticker's prices
    adjusted_cache = {}
    for ticker, rows in ticker_data.items():
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        cum_factor = 1.0
        adj_opens = []
        adj_highs = []
        adj_lows = []
        adj_closes = []
        
        for idx_row, row in df.iterrows():
            adj_opens.append(row['open'] * cum_factor)
            adj_highs.append(row['high'] * cum_factor)
            adj_lows.append(row['low'] * cum_factor)
            adj_closes.append(row['close'] * cum_factor)
            
            if row['adj_factor'] != 1.0:
                cum_factor *= row['adj_factor']
                
        df['open'] = adj_opens
        df['high'] = adj_highs
        df['low'] = adj_lows
        df['close'] = adj_closes
        
        df = df.sort_values('date').set_index('date')
        adjusted_cache[ticker] = df
        
    return adjusted_cache

print("Loading adjusted txt prices...")
adj_cache = load_adjusted_txt_prices(txt_path, isins_path)
print(f"Loaded {len(adj_cache)} tickers from txt.")

# Load original cache
print("Loading original cache...")
data = np.load(str(cache_path))
meta = pd.read_csv(cache_path.parent / "cache_meta.csv")

tk = "SAN"
print(f"\nTesting ticker: {tk}")
idx = pd.DatetimeIndex(data[f"idx_{tk}"])
c_orig = data[f"c_{tk}"]
print(f"Original hourly size: {len(c_orig)}")

# Map daily to hourly
df_daily = adj_cache[tk]
# Reindex with ffill
df_reindexed = df_daily.reindex(idx, method='ffill')
print(f"Reindexed daily size: {len(df_reindexed)}")
print("First few rows of comparison:")
comparison = pd.DataFrame({
    'orig': c_orig,
    'new_adj': df_reindexed['close']
}, index=idx)
print(comparison.head(10))
print("\nLast few rows of comparison:")
print(comparison.tail(10))
print("\nCheck if there are any NaNs after ffill:")
print(comparison.isna().sum())
