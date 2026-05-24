"""
Diagnóstico: Comparativa de factores que explican la divergencia entre
Backtesting_Capital_Real (positivo) y Backtesting_Avanzado (negativo)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

BASE_DIR   = Path(__file__).parent.resolve()
ROOT       = BASE_DIR.parent
CACHE_OLD  = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"
CACHE_LIVE = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"
TXT_PATHS  = [
    BASE_DIR / "HISTORICOS" / "Historicos IBEX 35 19052026.txt",
    BASE_DIR / "HISTORICOS" / "Historicos mercado continuo 19052026.txt"
]
ISIN_PATHS = [ROOT / "isins_ibex35.csv", ROOT / "isins_mercadocontinuo.csv"]

PROB_UMBRAL = 0.93
SL_PCT      = 0.047
TS_PCT      = 0.055
TARGET_LEN  = 60

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
                    h = float(parts[4]); l = float(parts[5]); c = float(parts[6])
                    if ticker not in ticker_data: ticker_data[ticker] = []
                    ticker_data[ticker].append({'date': parts[2].strip(), 'high': h, 'low': l, 'close': c})
                except: continue
                
    result = {}
    for ticker, rows in ticker_data.items():
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date').drop_duplicates('date').set_index('date')
        result[ticker] = df
        
    return result

# ── 1. Análisis: ¿Cuántas señales generan ambas caches? ──────────────────────
print("="*65)
print("DIAGNÓSTICO: COMPARATIVA CACHE_OLD vs CACHE_LIVE")
print("="*65)

txt_prices = load_txt_prices(TXT_PATHS, ISIN_PATHS)

results = []
for label, path in [("CACHE_OLD (Backtesting_Capital_Real)", CACHE_OLD),
                    ("CACHE_LIVE (Backtesting_Avanzado)",   CACHE_LIVE)]:
    if not path.exists():
        print(f"\n{label}: archivo no encontrado en {path}")
        continue
    data = np.load(str(path))
    tickers = sorted([k.replace("preds_","") for k in data.keys() if k.startswith("preds_")])
    
    señales_total = 0
    señales_txt   = 0
    wins_txt      = 0
    losses_txt    = 0
    ev_pct_list   = []

    for tk in tickers:
        if f"preds_{tk}" not in data: continue
        preds = data[f"preds_{tk}"]
        idx   = pd.DatetimeIndex(data[f"idx_{tk}"])
        
        if tk in txt_prices:
            df_r = txt_prices[tk].reindex(idx, method='ffill').bfill()
            c = df_r['close'].values.astype(np.float32)
            l = df_r['low'].values.astype(np.float32)
            h = df_r['high'].values.astype(np.float32)
        else:
            c = data.get(f"c_{tk}", None)
            l = data.get(f"l_{tk}", None)
            h = data.get(f"h_{tk}", None)
            if c is None: continue
        
        n = len(c)
        for k in range(len(preds)):
            i = k + TARGET_LEN
            if i >= n - 49: continue
            prob = float(np.max(preds[k]))
            if prob < PROB_UMBRAL: continue
            clase = int(np.argmax(preds[k]))
            if clase != 1: continue  # ALC_CONT
            señales_total += 1
            
            if tk in txt_prices:
                señales_txt += 1
                # Simulate 48h exit
                sl    = c[i] * (1 - SL_PCT)
                exit_p = c[i]
                for j in range(i+1, min(i+49, n)):
                    if l[j] <= sl: exit_p = sl; break
                else: exit_p = c[min(i+48, n-1)]
                gpc = (exit_p - c[i]) / c[i]
                ev_pct_list.append(gpc)
                if gpc > 0: wins_txt += 1
                else: losses_txt += 1
    
    wr = wins_txt / señales_txt * 100 if señales_txt > 0 else 0
    ev = np.mean(ev_pct_list) * 100 if ev_pct_list else 0
    print(f"\n{label}")
    print(f"  Tickers:         {len(tickers)}")
    print(f"  Señales ALC_CONT >{PROB_UMBRAL*100:.0f}%: {señales_total:,}")
    print(f"  Señales con txt: {señales_txt:,}")
    print(f"  Win Rate (48h):  {wr:.2f}%")
    print(f"  EV% medio (48h): {ev:+.3f}%")

# ── 2. Análisis: Impacto del filtro de régimen ────────────────────────────────
print("\n" + "="*65)
print("ANÁLISIS: IMPACTO FILTRO RÉGIMEN BULL/BEAR (sobre CACHE_LIVE)")
print("="*65)

ibex = yf.download("^IBEX", start="2017-01-01", progress=False)
if isinstance(ibex.columns, pd.MultiIndex):
    ibex_close = ibex['Close']['^IBEX']
else:
    ibex_close = ibex['Close']
sma200 = ibex_close.rolling(200).mean()
ibex_regime = ibex_close > sma200

data = np.load(str(CACHE_LIVE))
tickers = sorted([k.replace("preds_","") for k in data.keys() if k.startswith("preds_")])

bull_wins = bull_losses = bear_wins = bear_losses = 0
for tk in tickers:
    if tk not in txt_prices: continue
    preds = data[f"preds_{tk}"]
    idx   = pd.DatetimeIndex(data[f"idx_{tk}"])
    df_r  = txt_prices[tk].reindex(idx, method='ffill').bfill()
    c = df_r['close'].values.astype(np.float32)
    l = df_r['low'].values.astype(np.float32)
    n = len(c)
    
    for k in range(len(preds)):
        i = k + TARGET_LEN
        if i >= n - 49: continue
        prob = float(np.max(preds[k]))
        if prob < PROB_UMBRAL or int(np.argmax(preds[k])) != 1: continue
        
        fecha_d = idx[i].floor('D')
        try: is_bull = bool(ibex_regime.asof(fecha_d))
        except: is_bull = False
        
        sl = c[i] * (1 - SL_PCT)
        exit_p = c[i]
        for j in range(i+1, min(i+49, n)):
            if l[j] <= sl: exit_p = sl; break
        else: exit_p = c[min(i+48, n-1)]
        gpc = (exit_p - c[i]) / c[i]
        
        if is_bull:
            if gpc > 0: bull_wins += 1
            else: bull_losses += 1
        else:
            if gpc > 0: bear_wins += 1
            else: bear_losses += 1

bt = bull_wins + bull_losses
be = bear_wins + bear_losses
print(f"\n  Señales en régimen BULL: {bt:,}")
print(f"    Win Rate: {bull_wins/bt*100:.1f}%  ({bull_wins} ganan / {bull_losses} pierden)" if bt else "    Sin datos")
print(f"\n  Señales en régimen BEAR: {be:,}")
print(f"    Win Rate: {bear_wins/be*100:.1f}%  ({bear_wins} ganan / {bear_losses} pierden)" if be else "    Sin datos")
print()
