import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# --- CONFIGURACIÓN ALGORÍTMICA FINAL ---
CAPITAL_TOTAL = 10000.0  
PROB_UMBRAL = 0.93       
TOP_N = 8               # Seleccionamos solo los 8 mejores para la cartera
CACHE_PATH = Path("cache_predicciones_LIVE.npz")
REPORT_PATH = Path("README.md")

RULES = {
    "BULL": {"SL": "Trailing Stop 5.5%", "EXIT": "Tendencia (Largo Plazo)", "COLOR": "🟢"},
    "BEAR": {"SL": "Fijo 4.7%", "EXIT": "Cierre forzado a las 48h", "COLOR": "🔴"}
}

def get_market_info():
    try:
        ibex = yf.download("^IBEX", period="2y", progress=False)
        close = ibex["Close"]
        sma200 = close.rolling(window=200).mean().iloc[-1]
        last = close.iloc[-1]
        regime = "BULL" if last > sma200 else "BEAR"
        return regime, last, sma200
    except: return "BULL", 0, 0

def get_tier_info(tk, price, volume):
    vol_efectivo = price * volume
    if vol_efectivo > 40_000_000: return 1, 0.15 
    if vol_efectivo > 10_000_000: return 2, 0.08 
    return 3, 0.04                               

def main():
    if not CACHE_PATH.exists(): return
    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    regime, ibex_p, sma_p = get_market_info()
    current_rules = RULES[regime]
    
    all_signals = []
    for tk in tickers:
        preds = data[f"preds_{tk}"]
        idx = pd.to_datetime(data[f"idx_{tk}"])
        c, v = data[f"c_{tk}"], data[f"v_{tk}"]
        last_c, last_v = c[-1], v[-1]
        last_pred = preds[-1]
        clase = int(np.argmax(last_pred))
        prob = float(last_pred[clase])
        
        # CHEQUEO DE CALIDAD AUTOMÁTICO
        is_fresh = (datetime.now() - idx[-1]).total_seconds()/3600 < 48
        
        if clase == 1 and prob >= PROB_UMBRAL and is_fresh:
            tier, alloc = get_tier_info(tk, last_c, last_v)
            all_signals.append({
                "tk": tk, "prob": prob, "inv": CAPITAL_TOTAL * alloc,
                "price": last_c, "tier": tier, "date": idx[-1]
            })

    # ORDENAR POR CONFIANZA Y QUEDARNOS CON LOS MEJORES TOP_N
    top_signals = sorted(all_signals, key=lambda x: x['prob'], reverse=True)[:TOP_N]

    md = [
        f"# 🤖 Sistema de Trading IA - Dashboard TFM\n",
        f"**Estado:** `MERCADO ABIERTO` | **Régimen:** {current_rules['COLOR']} {regime}\n",
        f"## 🎯 CARTERA RECOMENDADA PARA HOY",
        f"| Ticker | Orden | Inversión | Stop Loss | Confianza | Último Dato |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for s in top_signals:
        sl_desc = "TS 5.5%" if regime == "BULL" else f"Fijo: {(s['price']*0.953):.3f}"
        md.append(f"| **{s['tk']}** | 🔵 COMPRAR | **{s['inv']:,.2f}€** | {sl_desc} | `{s['prob']*100:.1f}%` | {s['date'].strftime('%H:%M')} |")

    if not top_signals:
        md.append("| - | *Sin señales de alta confianza hoy* | - | - | - | - |")

    md.append("\n## 📋 Instrucciones de Operativa (Sin Selección Manual)")
    md.append(f"1. **Ejecución**: Comprar los activos de la tabla anterior con los importes indicados.")
    md.append(f"2. **Gestión de Salida**: Aplicar `{current_rules['SL']}`. Salida por tiempo: `{current_rules['EXIT']}`.")
    md.append(f"3. **Filtros Aplicados**: Solo se muestran activos con datos frescos (<48h) y Confianza > {PROB_UMBRAL*100}%.")

    md.append(f"\n---\n## 🛠️ Health Check (Estado de la Red)")
    md.append("| Ticker | Estado | Velas |")
    md.append("| :--- | :--- | :--- |")
    for tk in tickers:
        idx = pd.to_datetime(data[f"idx_{tk}"])
        q = "🟢 OK" if (datetime.now() - idx[-1]).total_seconds()/3600 < 48 else "🔴 ERROR"
        md.append(f"| {tk} | {q} | {len(idx)} |")

    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write("\n".join(md))

if __name__ == "__main__": main()


