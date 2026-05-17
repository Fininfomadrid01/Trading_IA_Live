"""
BACKTESTING CON GESTIÓN MONETARIA AVANZADA (TFM)
=================================================
Simulación integrando los 3 planos de análisis:
1. Plano Señal: Probabilidades de red neuronal
2. Plano Operación: Esperanza Matemática (EV%)
3. Plano Cartera: Curva de capital, Sharpe.

Gestión monetaria dinámica:
- Tiers de Liquidez (T1: 10%, T2: 4%, T3: 2%) rebalanceados
  trimestralmente usando el volumen mediano del trimestre anterior.
- Anti-Martingala Calibrada por Tier:
  Multiplicador tras TP = 1.17x, tras SL = 0.96x.
  Rango: [0.48, 1.25]
- Filtro de riesgo estructural (min_struct_risk) >= 4.7% (vía SL).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

CACHE_DIR = Path("C:/Users/User/Desktop/VALIDAR HISTORICOS/evaluacion_algoritmos/resultados/cache_predicciones_fixed")
OUT_DIR   = Path("C:/Users/User/Desktop/VALIDAR HISTORICOS/ENTREGA_TFM_VERSION_FINAL/resultados")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuración Estratégica ───────────────────────────────────────────────
PROB_UMBRAL  = 0.93
HORIZONTE_H  = 48
SL_PCT       = 0.047   # Adaptado a min_struct_risk = 4.7%
COOLDOWN_DIA = 3
TARGET_LEN   = 60
CAPITAL      = 100_000.0

# ── Configuración Gestión Monetaria ─────────────────────────────────────────
# Ajustado para reducir el Cash Drag manteniendo la proporcionalidad de liquidez
TIER_BASE_ALLOC = {1: 0.15, 2: 0.08, 3: 0.04}
AM_WIN_SCALE    = 1.17
AM_LOSS_SCALE   = 0.96
AM_MIN_MULT     = 0.50
AM_MAX_MULT     = 2.00

SUPER_LABELS = {0:"NONE",1:"ALC_CONT",2:"BAJ_CONT",3:"REV_ALCISTA",4:"REV_BAJISTA",5:"LATERAL"}

IB_PCT = 0.0005
IB_MIN = 3.00
def comision(pos): return 2 * max(pos * IB_PCT, IB_MIN)

# ─────────────────────────────────────────────────────────────────────────────

def calcular_volumenes_trimestrales(cache):
    """Calcula el volumen efectivo diario para cada ticker y lo agrupa por trimestre."""
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
    
    if not dfs:
        return pd.DataFrame()
        
    df_all = pd.concat(dfs)
    
    # Agrupar por trimestre y ticker, calculando la mediana del volumen diario
    vol_q = df_all.groupby(['quarter', 'tk'])['vol_efectivo'].median().reset_index()
    print("OK")
    return vol_q

def obtener_tier(tk, current_quarter, vol_q):
    """Obtiene el Tier de un ticker basándose en el volumen del trimestre anterior."""
    prev_quarter = current_quarter - 1
    vols_prev = vol_q[vol_q['quarter'] == prev_quarter]
    
    if vols_prev.empty:
        return 3 # Default si no hay datos
        
    vols_prev = vols_prev.sort_values('vol_efectivo', ascending=False).reset_index(drop=True)
    n = len(vols_prev)
    if n == 0:
        return 3
        
    t1_limit = n // 3
    t2_limit = (2 * n) // 3
    
    # Encontrar la posición del ticker
    pos = vols_prev[vols_prev['tk'] == tk].index
    if len(pos) == 0:
        return 3
        
    idx = pos[0]
    if idx < t1_limit: return 1
    elif idx < t2_limit: return 2
    else: return 3

# ─────────────────────────────────────────────────────────────────────────────

def fase1_generar_trades_potenciales(cache):
    """Genera las señales brutas con los precios de entrada y salida exactos."""
    cooldown_td = pd.Timedelta(days=COOLDOWN_DIA)
    trades = []

    for tk in sorted(cache.keys()):
        preds = cache[tk]["preds"]
        idx   = cache[tk]["idx"]
        c     = cache[tk]["c"]
        l     = cache[tk]["l"]
        n     = len(c)

        ultima_señal = None

        for k in range(len(preds)):
            i = k + TARGET_LEN
            if i + HORIZONTE_H >= n: continue

            if ultima_señal is not None and (idx[i] - ultima_señal) < cooldown_td:
                continue

            clase = int(np.argmax(preds[k]))
            prob  = float(np.max(preds[k]))
            if prob < PROB_UMBRAL or SUPER_LABELS[clase] != "ALC_CONT":
                continue

            # Filtro de fecha TFM
            if idx[i] < pd.Timestamp("2018-04-24"):
                continue

            entry_price = c[i]
            sl_price    = entry_price * (1 - SL_PCT)
            exit_i      = i + HORIZONTE_H
            exit_reason = "TIEMPO"

            for j in range(i + 1, min(i + HORIZONTE_H + 1, n)):
                if l[j] <= sl_price:
                    exit_i      = j
                    exit_reason = "SL"
                    break

            exit_price = sl_price if exit_reason == "SL" else c[exit_i]
            
            # Nota: PnL y posición se calcularán en Fase 2 (dinámicamente)
            trades.append({
                "ticker"      : tk,
                "entry_ts"    : idx[i],
                "exit_ts"     : idx[exit_i],
                "entry_price" : round(float(entry_price), 4),
                "exit_price"  : round(float(exit_price), 4),
                "motivo"      : exit_reason,
                "prob"        : round(prob, 4),
                "ganancia_pct": (exit_price - entry_price) / entry_price
            })
            ultima_señal = idx[i]

    # Ordenar cronológicamente y por probabilidad
    return pd.DataFrame(trades).sort_values(
        ["entry_ts", "prob"], ascending=[True, False]
    ).reset_index(drop=True)

def fase2_aplicar_gestion_monetaria(df_pot, vol_q):
    """
    Aplica MM Avanzada:
    1. Calcula el Tier dinámico basado en Q-1.
    2. Aplica multiplicador Anti-Martingala por Tier.
    3. Reinvierte capital compuesto.
    """
    capital_cash   = CAPITAL
    equity_realized = CAPITAL
    posiciones     = {}   
    last_exit_tk   = {}   
    ejecutadas     = []
    cooldown_td    = pd.Timedelta(days=COOLDOWN_DIA)
    
    # Estado Anti-Martingala por Tier
    am_multipliers = {1: 1.0, 2: 1.0, 3: 1.0}

    for _, row in df_pot.iterrows():
        entry_ts = row["entry_ts"]
        tk       = row["ticker"]
        current_q = entry_ts.to_period('Q')

        # 1. Cerrar posiciones que ya han terminado
        cerradas = [t for t, p in posiciones.items() if p["exit_ts"] <= entry_ts]
        for t in cerradas:
            p = posiciones.pop(t)
            capital_cash += p["pos_eur"] + p["pnl_neto"]
            equity_realized += p["pnl_neto"]
            last_exit_tk[t] = p["exit_ts"]
            
            # Actualizar Anti-Martingala según el resultado
            tier_trade = p["tier"]
            if p["ganancia_pct"] > 0:
                am_multipliers[tier_trade] = min(AM_MAX_MULT, am_multipliers[tier_trade] * AM_WIN_SCALE)
            else:
                am_multipliers[tier_trade] = max(AM_MIN_MULT, am_multipliers[tier_trade] * AM_LOSS_SCALE)

        if tk in posiciones:
            continue

        if tk in last_exit_tk:
            if (entry_ts - last_exit_tk[tk]) < cooldown_td:
                continue

        # Calcular Asignación Efectiva
        tier = obtener_tier(tk, current_q, vol_q)
        base_alloc = TIER_BASE_ALLOC[tier]
        eff_alloc  = base_alloc * am_multipliers[tier]
        
        # Sizing basado en EQUITY TOTAL (Caja + Valor invertido aproximado), no solo en la caja restante
        current_equity = capital_cash + sum(p["pos_eur"] for p in posiciones.values())
        pos_eur = current_equity * eff_alloc
        
        if capital_cash < pos_eur or pos_eur < 500: # Rechazar si no hay caja suficiente
            continue

        # Ejecutar operación
        com = comision(pos_eur)
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
        
        # Registrar trade
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

    # Cerrar remanentes al final
    for t, p in posiciones.items():
        capital_cash += p["pos_eur"] + p["pnl_neto"]
        equity_realized += p["pnl_neto"]

    return pd.DataFrame(ejecutadas).reset_index(drop=True), capital_cash

# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("BACKTESTING GESTIÓN MONETARIA TFM (ANTI-MARTINGALA & TIERS)")
    print("=" * 70)
    print(f"  Capital Inicial: {CAPITAL:,.0f} EUR")
    print(f"  Tiers Alloc:     T1 10%, T2 4%, T3 2% (Rebalanceo Trimestral Lagged)")
    print(f"  Anti-Martingala: TP x{AM_WIN_SCALE}, SL x{AM_LOSS_SCALE} | Limites [{AM_MIN_MULT}, {AM_MAX_MULT}]")
    print(f"  Structural Risk: min_struct_risk >= {SL_PCT*100}% (SL)")
    print(f"  Filtro Fecha:    >= 2018-04-24\n")

    print("  Cargando cache de predicciones...", end=" ", flush=True)
    data = np.load(str(CACHE_DIR / "cache_predicciones.npz"), allow_pickle=False)
    meta = pd.read_csv(CACHE_DIR / "cache_meta.csv")
    cache = {}
    for tk in sorted(meta["ticker"].tolist()):
        try:
            cache[tk] = dict(
                preds=data[f"preds_{tk}"].astype(np.float32),
                idx=pd.DatetimeIndex(data[f"idx_{tk}"]),
                c=data[f"c_{tk}"], l=data[f"l_{tk}"], v=data[f"v_{tk}"]
            )
        except: pass
    print(f"OK ({len(cache)} tickers)")
    
    vol_q = calcular_volumenes_trimestrales(cache)

    print(f"  Fase 1: generando señales potenciales...", end=" ", flush=True)
    df_pot = fase1_generar_trades_potenciales(cache)
    print(f"OK  ({len(df_pot):,} señales)")

    print(f"  Fase 2: aplicando MM dinámica...", end=" ", flush=True)
    df_exec, capital_final = fase2_aplicar_gestion_monetaria(df_pot, vol_q)
    print(f"OK  ({len(df_exec):,} ejecutadas)")

    if len(df_exec) == 0:
        print("\n  Sin operaciones ejecutadas."); return

    # ── Métricas ─────────────────────────────────────────────────────────
    ganadas  = df_exec[df_exec["pnl_neto"] > 0]
    perdidas = df_exec[df_exec["pnl_neto"] <= 0]
    wr       = len(ganadas) / len(df_exec) if len(df_exec) > 0 else 0
    
    avg_win_pct  = ganadas["ganancia_pct"].mean() if len(ganadas) > 0 else 0
    avg_loss_pct = perdidas["ganancia_pct"].mean() if len(perdidas) > 0 else 0
    
    # EV = WR * AvgWin + (1-WR) * AvgLoss
    ev_pct = (wr * avg_win_pct) + ((1 - wr) * avg_loss_pct)

    pnl_total    = df_exec["pnl_neto"].sum()
    roi          = pnl_total / CAPITAL * 100
    años         = (df_exec["exit_ts"].max() - df_exec["entry_ts"].min()).days / 365.25
    roi_anual    = ((capital_final/CAPITAL)**(1/años) - 1)*100 if años > 0 else 0

    equity = CAPITAL + df_exec.sort_values("exit_ts")["pnl_neto"].cumsum().values
    peak   = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1).min() * 100)
    
    # Aproximación básica de Sharpe (retorno anualizado vs volatilidad equity curva)
    daily_returns = pd.Series(equity).pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 else 0

    print(f"\n{'='*70}")
    print(f"PLANO 2: RENDIMIENTO DE LA SEÑAL (ESPERANZA MATEMÁTICA)")
    print(f"{'='*70}")
    print(f"  Win Rate (WR):            {wr*100:>7.2f}%")
    print(f"  Avg Win %:                {avg_win_pct*100:>+7.2f}%")
    print(f"  Avg Loss %:               {avg_loss_pct*100:>+7.2f}%")
    print(f"  Esperanza (EV%):          {ev_pct*100:>+7.3f}%  (por operación)")

    print(f"\n{'='*70}")
    print(f"PLANO 3: RENDIMIENTO DE LA CARTERA (EQUITY Y RIESGO)")
    print(f"{'='*70}")
    print(f"  Capital inicial:          {CAPITAL:>12,.2f} EUR")
    print(f"  Capital final:            {capital_final:>12,.2f} EUR")
    print(f"  Beneficio neto:           {pnl_total:>+12,.2f} EUR")
    print(f"  ROI neto total:           {roi:>+11.2f}%")
    print(f"  ROI anualizado:           {roi_anual:>+11.2f}%")
    print(f"  Max Drawdown:             {max_dd:>+11.2f}%")
    print(f"  Ratio Sharpe (Aprox):     {sharpe:>11.2f}")
    print()
    print(f"  Operaciones Totales:      {len(df_exec):>8,}")
    print(f"  Distribución Tiers:       T1: {sum(df_exec['tier']==1)}, T2: {sum(df_exec['tier']==2)}, T3: {sum(df_exec['tier']==3)}")

    df_exec.to_csv(OUT_DIR / "Resultados_MM_Avanzada.csv", index=False)
    print(f"\n  Archivo: {OUT_DIR / 'Resultados_MM_Avanzada.csv'}")
    print(f"  Tiempo total: {(time.time()-t0)/60:.1f} min")
    print("\n  COMPLETADO")

if __name__ == "__main__":
    main()
