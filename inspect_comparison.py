import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS")
cache_path = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"
txt_path = ROOT / "ENTREGA_TFM_VERSION_FINAL" / "HISTORICOS" / "Historicos IBEX 35 19052026.txt"

# Load ISIN mapping
ibex_isins = pd.read_csv(ROOT / "isins_ibex35.csv")
ibex_isins = ibex_isins.rename(columns={'CVALISO': 'isin'})
ibex_isins['isin'] = ibex_isins['isin'].astype(str).str.strip()
ibex_isins['VALOR'] = ibex_isins['VALOR'].astype(str).str.strip()
isin_to_ticker = dict(zip(ibex_isins['isin'], ibex_isins['VALOR']))

# Let's read some lines for SAN (ISIN ES0113900J37) from txt
san_rows = []
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
                    if isin == "ES0113900J37":  # SAN
                        date = parts[2].strip()
                        o = float(parts[3].strip())
                        h = float(parts[4].strip())
                        l = float(parts[5].strip())
                        c = float(parts[6].strip())
                        adj = float(parts[9].strip())
                        v = float(parts[11].strip())
                        san_rows.append({
                            'date': date, 'open': o, 'high': h, 'low': l, 'close': c, 'adj': adj, 'volume': v
                        })

df_txt = pd.DataFrame(san_rows)
df_txt['date'] = pd.to_datetime(df_txt['date'], format='%Y%m%d')
df_txt = df_txt.sort_values('date').reset_index(drop=True)

print("SAN daily data from TXT (head):")
print(df_txt.head(10))
print("\nSAN daily data from TXT (tail):")
print(df_txt.tail(10))

# Load SAN from cache
if cache_path.exists():
    data = np.load(str(cache_path))
    idx = pd.DatetimeIndex(data["idx_SAN"])
    c = data["c_SAN"]
    l = data["l_SAN"]
    h = data["h_SAN"]
    v = data["v_SAN"]
    df_cache = pd.DataFrame({'close': c, 'low': l, 'high': h, 'volume': v}, index=idx)
    # Resample cache to daily to compare
    df_cache_daily = df_cache.resample('1D').last().dropna()
    print("\nSAN daily data from cache (tail):")
    print(df_cache_daily.tail(10))
    
    # Compare matching dates
    merged = pd.merge(df_txt, df_cache_daily, left_on='date', right_index=True, suffixes=('_txt', '_cache'))
    print("\nMerged comparison (tail):")
    print(merged[['date', 'close_txt', 'close_cache', 'adj']].tail(10))
else:
    print("Cache not found.")
