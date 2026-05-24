import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS")
cache_path = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"
txt_path = ROOT / "ENTREGA_TFM_VERSION_FINAL" / "HISTORICOS" / "Historicos IBEX 35 19052026.txt"

# Let's read IBE (ISIN ES0144580Y14) from txt
rows = []
with open(txt_path, 'r', encoding='latin1') as f:
    for line in f:
        if line.startswith("0001"):
            parts = line.split(",")
            if len(parts) > 11:
                raw_code = parts[1].strip()
                subparts = [p for p in raw_code.split() if p]
                if subparts:
                    isin = subparts[0]
                    if isin.startswith("XX") or isin.startswith("YY") or isin.startswith("ZZ"):
                        isin = isin[2:]
                    if isin == "ES0144580Y14":  # IBE
                        date = parts[2].strip()
                        o = float(parts[3].strip())
                        h = float(parts[4].strip())
                        l = float(parts[5].strip())
                        c = float(parts[6].strip())
                        adj = float(parts[9].strip())
                        v = float(parts[11].strip())
                        rows.append({
                            'date': date, 'open': o, 'high': h, 'low': l, 'close': c, 'adj_factor': adj, 'volume': v
                        })

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values('date', ascending=False).reset_index(drop=True) # Descending for adjustment

# Apply cumulative adjustment backwards
cum_factor = 1.0
adj_closes = []
adj_highs = []
adj_lows = []
adj_opens = []

for idx_row, row in df.iterrows():
    # Apply factor
    adj_opens.append(row['open'] * cum_factor)
    adj_highs.append(row['high'] * cum_factor)
    adj_lows.append(row['low'] * cum_factor)
    adj_closes.append(row['close'] * cum_factor)
    
    # Update factor if there was an adjustment on this day
    if row['adj_factor'] != 1.0:
        cum_factor *= row['adj_factor']

df['adj_open'] = adj_opens
df['adj_high'] = adj_highs
df['adj_low'] = adj_lows
df['adj_close'] = adj_closes

# Sort ascending for comparison
df = df.sort_values('date').reset_index(drop=True)

print("Adjusted IBE daily data from TXT (head):")
print(df[['date', 'close', 'adj_factor', 'adj_close']].head(10))
print("\nAdjusted IBE daily data from TXT (tail):")
print(df[['date', 'close', 'adj_factor', 'adj_close']].tail(10))

# Load IBE from cache to compare
if cache_path.exists():
    data = np.load(str(cache_path))
    idx = pd.DatetimeIndex(data["idx_IBE"])
    c = data["c_IBE"]
    df_cache = pd.DataFrame({'close_cache': c}, index=idx)
    df_cache_daily = df_cache.resample('1D').last().dropna()
    
    # Merge and compare
    merged = pd.merge(df, df_cache_daily, left_on='date', right_index=True)
    print("\nComparison of Adjusted Close (TXT) vs Cache Close (Yahoo):")
    print(merged[['date', 'close', 'adj_close', 'close_cache']].tail(20))
    # Print correlation or mean absolute percentage error
    mape = np.mean(np.abs(merged['adj_close'] - merged['close_cache']) / merged['close_cache']) * 100
    print(f"\nMean Absolute Percentage Error between TXT-adjusted and Yahoo: {mape:.2f}%")
else:
    print("Cache not found.")
