import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# --- CONFIGURACIÓN TFM (IA REGIME SWITCHING) ---
CAPITAL_TOTAL = 10000.0  
PROB_UMBRAL = 0.93       
CACHE_PATH = Path("cache_predicciones_LIVE.npz")
REPORT_PATH = Path("README.md")

RULES = {
    "BULL": {"SL": "Trailing Stop 5.5%", "EXIT": "Tendencia", "COLOR": "🟢"},
    "BEAR": {"SL": "Fijo 4.7%", "EXIT": "48h", "COLOR": "🔴"}
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
    now_str = datetime.now().strftime('%d/%m/%Y %H:%M')
    
    md = [
        f"# 🤖 IA Propuesta: Regime Switching (TFM)\n",
        f"**Estado:** `ESTRATEGIA ACTIVA` | **Actualizado:** `{now_str} UTC`\n",
        f"## 📊 Análisis de Régimen ({current_rules['COLOR']} {regime})",
        f"- **IBEX 35:** {ibex_p:.2f} | **SMA200:** {sma_p:.2f}",
        f"- **Configuración:** {current_rules['SL']} | Salida: {current_rules['EXIT']}\n",
        f"---",
        f"## 🎯 ORDENES OPERATIVAS PARA HOY",
        f"| Ticker | Orden | Inversión (EUR) | Tier | Stop Loss | Confianza |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for tk in tickers:
        preds = data[f"preds_{tk}"]
        idx = pd.to_datetime(data[f"idx_{tk}"])
        c, v = data[f"c_{tk}"], data[f"v_{tk}"]
        last_c, last_v = c[-1], v[-1]
        last_pred = preds[-1]
        
        # Corrección del error anterior: asignación en dos pasos
        clase = int(np.argmax(last_pred))
        prob = float(last_pred[clase])
        
        if clase == 1 and prob >= PROB_UMBRAL:
            tier, alloc = get_tier_info(tk, last_c, last_v)
            inv_eur = CAPITAL_TOTAL * alloc
            sl_desc = "TS 5.5%" if regime == "BULL" else f"Fijo: {(last_c*0.953):.3f}"
            md.append(f"| **{tk}** | 🔵 COMPRAR | **{inv_eur:,.2f}€** | T{tier} | {sl_desc} | `{prob*100:.1f}%` |")

    md.append("\n## 📒 Bitácora de Gestión Monetaria")
    md.append(f"- **Capital Base:** {CAPITAL_TOTAL:,.0f}€")
    md.append(f"- **Reglas:** Gestión por Tiers (15%/8%/4%) y Salida Asimétrica (Regime Switching).")

    md.append(f"\n---\n*Dashboard automatizado bajo la arquitectura final del TFM: CNN-BiLSTM + Regime Switching.*")

    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write("\n".join(md))

if __name__ == "__main__": main()
