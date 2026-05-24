"""
BACKTESTING AVANZADO (TFM EXTRA)
================================
Versión experimental con:
1. Operativa Solo Largos (Long-Only).
2. Trailing Stop Dinámico (3.5%) sin límite estricto de tiempo (máx 150 días).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import yfinance as yf

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"
OUT_DIR   = BASE_DIR / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuración Estratégica Avanzada ──────────────────────────────────────
PROB_UMBRAL       = 0.93
SL_PCT            = 0.047   
TS_PCT            = 0.055   
COOLDOWN_DIA      = 3
CAPITAL           = 100_000.0

# Definición de Regímenes (IBEX vs SMA200)
# Bull Regime: MAX_HOLD = 150, Trailing Stop activo
# Bear Regime: MAX_HOLD = 48,  Salida fija por tiempo


# Gestión Monetaria (Perfil Realista-Agresivo)
TIER_BASE_ALLOC = {1: 0.15, 2: 0.08, 3: 0.04}
AM_WIN_SCALE    = 1.17
AM_LOSS_SCALE   = 0.96
AM_MIN_MULT     = 0.50
AM_MAX_MULT     = 2.00

SUPER_LABELS = {0:"NONE",1:"ALC_CONT",2:"BAJ_CONT",3:"REV_ALCISTA",4:"REV_BAJISTA",5:"LATERAL"}

IB_PCT = 0.0005
IB_MIN = 1.25 # Unificado con el resto
SLIPPAGE = 0.0002
def comision(pos): return 2 * max(pos * (IB_PCT + SLIPPAGE), IB_MIN)

# ─────────────────────────────────────────────────────────────────────────────

def calcular_volumenes_trimestrales(cache):
    print("  Calculando volúmenes trimestrales para Tiers...", end=" ", flush=True)
    dfs = []
    for tk, data in cache.items():
        c = data["c"]
        v = data["v"]
        idx = data["idx"]
        df_tk = pd.DataFrame({'tk': tk, 'vol_efectivo': c * v}, index=idx)
        df_daily = df_tk.resample('1D').agg({'vol_efectivo': 'sum'}).dropna()
        df_daily['tk'] = tk
        df_daily = df_daily[df_daily['vol_efectivo'] > 0]
        if not df_daily.empty:
            df_daily['quarter'] = df_daily.index.to_period('Q')
            dfs.append(df_daily)
    
    if not dfs: return pd.DataFrame()
    df_all = pd.concat(dfs)
    vol_q = df_all.groupby(['quarter', 'tk'])['vol_efectivo'].median().reset_index()
    print("OK")
    return vol_q

def obtener_tier(tk, current_quarter, vol_q):
    prev_quarter = current_quarter - 1
    vols_prev = vol_q[vol_q['quarter'] == prev_quarter]
    if vols_prev.empty: return 3
    vols_prev = vols_prev.sort_values('vol_efectivo', ascending=False).reset_index(drop=True)
    n = len(vols_prev)
    if n == 0: return 3
    t1_limit = n // 3
    t2_limit = (2 * n) // 3
    pos = vols_prev[vols_prev['tk'] == tk].index
    if len(pos) == 0: return 3
    idx = pos[0]
    if idx < t1_limit: return 1
    elif idx < t2_limit: return 2
    else: return 3

# ─────────────────────────────────────────────────────────────────────────────

def fase1_generar_trades_potenciales(cache, ibex_regime):
    cooldown_td = pd.Timedelta(days=COOLDOWN_DIA)
    trades = []

    for tk in sorted(cache.keys()):
        preds = cache[tk]["preds"]
        idx   = cache[tk]["idx"]
        c     = cache[tk]["c"]
        l     = cache[tk]["l"]
        h     = cache[tk].get("h", c) 
        n     = len(c)

        ultima_señal = None
        TARGET_LEN = 60

        for k in range(len(preds)):
            i = k + TARGET_LEN
            if i >= n - 1: continue

            if ultima_señal is not None and (idx[i] - ultima_señal) < cooldown_td:
                continue

            clase = int(np.argmax(preds[k]))
            prob  = float(np.max(preds[k]))
            label = SUPER_LABELS[clase]
            
            if prob < PROB_UMBRAL or label != "ALC_CONT":
                continue

            if idx[i] < pd.Timestamp("2018-04-24"):
                continue

            # DETERMINAR RÉGIMEN EN EL MOMENTO DE ENTRADA
            fecha_d = idx[i].floor('D')
            if fecha_d in ibex_regime.index:
                is_bull = ibex_regime.loc[fecha_d]
            else:
                # Si no hay dato exacto (finde), buscar el anterior
                try: is_bull = ibex_regime.asof(fecha_d)
                except: is_bull = False

            entry_price = c[i]
            
            if is_bull:
                # REGIMEN ALCISTA: Paciencia + Trailing Stop
                exit_i      = min(i + 150, n - 1)
                exit_reason = "TIEMPO_MAX"
                sl_fijo     = entry_price * (1 - SL_PCT)
                max_visto   = entry_price
                stop_real   = sl_fijo
                
                for j in range(i + 1, exit_i + 1):
                    if h[j] > max_visto: max_visto = h[j]
                    ts_dinamico = max_visto * (1 - TS_PCT)
                    stop_real   = max(sl_fijo, ts_dinamico)
                    if l[j] <= stop_real:
                        exit_i = j
                        exit_reason = "TRAILING_STOP"
                        break
                exit_price = stop_real if "STOP" in exit_reason else c[exit_i]
            else:
                # REGIMEN BAJISTA/LATERAL: Rapidez (48h) + SL Fijo
                exit_i      = min(i + 48, n - 1)
                exit_reason = "TIEMPO_FIX"
                sl_fijo     = entry_price * (1 - SL_PCT)
                
                for j in range(i + 1, exit_i + 1):
                    if l[j] <= sl_fijo:
                        exit_i = j
                        exit_reason = "SL_FIJO"
                        break
                exit_price = sl_fijo if exit_reason == "SL_FIJO" else c[exit_i]

            ganancia_pct = (exit_price - entry_price) / entry_price

            trades.append({
                "ticker"      : tk,
                "tipo"        : "LONG",
                "regimen"     : "BULL" if is_bull else "BEAR",
                "entry_ts"    : idx[i],
                "exit_ts"     : idx[exit_i],
                "entry_price" : round(float(entry_price), 4),
                "exit_price"  : round(float(exit_price), 4),
                "motivo"      : exit_reason,
                "prob"        : round(prob, 4),
                "ganancia_pct": ganancia_pct
            })
            ultima_señal = idx[i]

    return pd.DataFrame(trades).sort_values(["entry_ts", "prob"], ascending=[True, False]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────

def fase2_aplicar_gestion_monetaria(df_pot, vol_q):
    capital_cash   = CAPITAL
    equity_realized = CAPITAL
    posiciones     = {}   
    last_exit_tk   = {}   
    ejecutadas     = []
    cooldown_td    = pd.Timedelta(days=COOLDOWN_DIA)
    
    am_multipliers = {1: 1.0, 2: 1.0, 3: 1.0}

    for _, row in df_pot.iterrows():
        entry_ts = row["entry_ts"]
        tk       = row["ticker"]
        current_q = entry_ts.to_period('Q')

        # Cerrar posiciones finalizadas
        cerradas = [t for t, p in posiciones.items() if p["exit_ts"] <= entry_ts]
        for t in cerradas:
            p = posiciones.pop(t)
            capital_cash += p["pos_eur"] + p["pnl_neto"]
            equity_realized += p["pnl_neto"]
            last_exit_tk[t] = p["exit_ts"]
            
            tier_trade = p["tier"]
            if p["ganancia_pct"] > 0:
                am_multipliers[tier_trade] = min(AM_MAX_MULT, am_multipliers[tier_trade] * AM_WIN_SCALE)
            else:
                am_multipliers[tier_trade] = max(AM_MIN_MULT, am_multipliers[tier_trade] * AM_LOSS_SCALE)

        if tk in posiciones: continue
        if tk in last_exit_tk and (entry_ts - last_exit_tk[tk]) < cooldown_td: continue

        tier = obtener_tier(tk, current_q, vol_q)
        base_alloc = TIER_BASE_ALLOC[tier]
        eff_alloc  = base_alloc * am_multipliers[tier]
        
        current_equity = capital_cash + sum(p["pos_eur"] for p in posiciones.values())
        pos_eur = current_equity * eff_alloc
        
        if capital_cash < pos_eur or pos_eur < 500: continue

        # Tanto Long como Short pagan la misma comision sobre volumen
        com = comision(pos_eur)
        # ganancia_pct ya esta resuelta direccionalmente en Fase 1
        pnl_bruto = pos_eur * row["ganancia_pct"] 
        pnl_neto  = pnl_bruto - com
        
        capital_cash -= pos_eur
        posiciones[tk] = {
            "exit_ts"     : row["exit_ts"],
            "pos_eur"     : pos_eur,
            "pnl_neto"    : pnl_neto,
            "ganancia_pct": row["ganancia_pct"],
            "tier"        : tier
        }
        
        res_row = row.to_dict()
        res_row["tier"] = tier
        res_row["eff_alloc_pct"] = round(eff_alloc * 100, 2)
        res_row["am_mult"] = round(am_multipliers[tier], 2)
        res_row["pos_eur"] = round(pos_eur, 2)
        res_row["pnl_bruto"] = round(float(pnl_bruto), 2)
        res_row["com_eur"] = round(com, 2)
        res_row["pnl_neto"] = round(float(pnl_neto), 2)
        res_row["capital_inicio"] = round(current_equity, 2)
        ejecutadas.append(res_row)

    for t, p in posiciones.items():
        capital_cash += p["pos_eur"] + p["pnl_neto"]
        equity_realized += p["pnl_neto"]

    return pd.DataFrame(ejecutadas).reset_index(drop=True), capital_cash

# ─────────────────────────────────────────────────────────────────────────────

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
                    o = float(parts[3].strip())
                    h = float(parts[4].strip())
                    l = float(parts[5].strip())
                    c = float(parts[6].strip())
                    v = float(parts[11].strip())
                    
                    if ticker not in ticker_data:
                        ticker_data[ticker] = []
                    ticker_data[ticker].append({
                        'date': date, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
                    })
                except: continue
                
    cache_prices = {}
    for ticker, rows in ticker_data.items():
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date').drop_duplicates('date').set_index('date')
        cache_prices[ticker] = df
        
    return cache_prices

def main():
    t0 = time.time()
    print("BACKTESTING AVANZADO (REGIME SWITCHING: BULL/BEAR)")
    print("=" * 60)

    # 1. Preparar Régimen del Mercado (IBEX 35)
    print("  Descargando datos IBEX 35 para análisis de régimen...", end=" ", flush=True)
    ibex = yf.download("^IBEX", start="2017-01-01", progress=False) 
    
    if isinstance(ibex.columns, pd.MultiIndex):
        ibex_close = ibex['Close']['^IBEX']
    else:
        ibex_close = ibex['Close']
        
    sma200 = ibex_close.rolling(window=200).mean()
    ibex_regime = ibex_close > sma200 # True = Bull, False = Bear
    print("OK")

    print("  Cargando precios locales de España (Historicos IBEX + Continuo)...", end=" ", flush=True)
    txt_paths = [
        BASE_DIR / "HISTORICOS" / "Historicos IBEX 35 19052026.txt",
        BASE_DIR / "HISTORICOS" / "Historicos mercado continuo 19052026.txt"
    ]
    isin_paths = [ROOT / "isins_ibex35.csv", ROOT / "isins_mercadocontinuo.csv"]
    txt_prices = load_txt_prices(txt_paths, isin_paths)
    print(f"OK ({len(txt_prices)} tickers cargados)")

    print(f"  Cargando cache: {CACHE_PATH.name}...")
    data = np.load(str(CACHE_PATH), allow_pickle=False)
    meta = pd.read_csv(CACHE_PATH.parent / "cache_meta.csv")
    cache = {}
    for tk in sorted(meta["ticker"].tolist()):
        try:
            idx = pd.DatetimeIndex(data[f"idx_{tk}"])
            preds = data[f"preds_{tk}"].astype(np.float32)
            
            if tk in txt_prices:
                df_tk = txt_prices[tk]
                df_reindexed = df_tk.reindex(idx, method='ffill').bfill()
                c = df_reindexed['close'].values.astype(np.float32)
                l = df_reindexed['low'].values.astype(np.float32)
                h = df_reindexed['high'].values.astype(np.float32)
                v = df_reindexed['volume'].values.astype(np.float32)
            else:
                c = data[f"c_{tk}"]
                l = data[f"l_{tk}"]
                h = data[f"h_{tk}"]
                v = data[f"v_{tk}"]
                
            cache[tk] = dict(
                preds=preds,
                idx=idx,
                c=c, l=l, h=h, v=v
            )
        except: pass
    
    vol_q = calcular_volumenes_trimestrales(cache)
    df_pot = fase1_generar_trades_potenciales(cache, ibex_regime)
    df_exec, cap_final = fase2_aplicar_gestion_monetaria(df_pot, vol_q)

    if len(df_exec) == 0:
        print("Sin operaciones."); return

    ganadas  = df_exec[df_exec["pnl_neto"] > 0]
    perdidas = df_exec[df_exec["pnl_neto"] <= 0]
    wr       = len(ganadas) / len(df_exec) if len(df_exec) > 0 else 0
    avg_w    = ganadas["ganancia_pct"].mean() if len(ganadas) > 0 else 0
    avg_l    = perdidas["ganancia_pct"].mean() if len(perdidas) > 0 else 0
    ev       = (wr * avg_w) + ((1-wr) * avg_l)

    pnl_total = df_exec["pnl_neto"].sum()
    roi       = pnl_total / CAPITAL * 100
    
    equity = CAPITAL + df_exec.sort_values("exit_ts")["pnl_neto"].cumsum().values
    peak   = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1).min() * 100)
    d_rets = pd.Series(equity).pct_change().dropna()
    sharpe = (d_rets.mean() / d_rets.std()) * np.sqrt(252) if len(d_rets)>0 else 0

    print(f"\nPLANO 2: RENDIMIENTO DE LA SEÑAL")
    print(f"  Win Rate (WR): {wr*100:.2f}% | EV%: {ev*100:+.3f}%")
    print(f"  Avg Win: {avg_w*100:+.2f}% | Avg Loss: {avg_l*100:+.2f}%")
    
    print(f"\nPLANO 3: RENDIMIENTO CARTERA AVANZADA")
    print(f"  Operaciones Totales: {len(df_exec)}")
    print(f"    - LONG:  {sum(df_exec['tipo']=='LONG')}")
    print(f"  Beneficio Neto: {pnl_total:>+12,.2f} EUR")
    print(f"  ROI Total:      {roi:>+11.2f}%")
    print(f"  Max Drawdown:   {max_dd:>+11.2f}%")
    print(f"  Ratio Sharpe:   {sharpe:>11.2f}")
    
    out_file = OUT_DIR / "Resultados_Avanzados.csv"
    df_exec.to_csv(out_file, index=False)
    print(f"\n  Guardado: {out_file}")

if __name__ == "__main__":
    main()
