import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# --- CONFIGURACIÓN ESTRATÉGICA (Ajusta tu capital aquí) ---
CAPITAL_TOTAL = 10000.0  # <--- Pon aquí tu capital real para calcular los euros
PROB_UMBRAL = 0.93
SL_PCT = 0.047           # 4.7% Stop Loss (Regla TFM)
TIER_ALLOC = {1: 0.15, 2: 0.08, 3: 0.04} # % de inversión según liquidez
CACHE_PATH = Path("cache_predicciones_LIVE.npz")
REPORT_PATH = Path("README.md")

def get_market_regime():
    try:
        # Versión robusta para el IBEX
        ibex = yf.download("^IBEX", period="2y", progress=False)
        close = ibex["Close"]
        sma200 = close.rolling(window=200).mean().iloc[-1]
        last = close.iloc[-1]
        return ("🟢 ALCISTA" if last > sma200 else "🔴 BAJISTA"), last, sma200
    except: return "Desconocido", 0, 0

def get_tier(tk, volume_median):
    # Lógica simplificada de Tiers para el reporte diario
    if volume_median > 50_000_000: return 1
    if volume_median > 10_000_000: return 2
    return 3

def main():
    if not CACHE_PATH.exists(): return
    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    regime, price, sma = get_market_regime()
    
    md = [
        f"# ⚡ Trading Desk IA - Inferencia en Vivo\n",
        f"**Estado del Sistema:** `OPERATIVO` | **Fecha:** `{datetime.now().strftime('%d/%m/%Y %H:%M')} UTC`\n",
        f"## 📊 Análisis de Entorno",
        f"- **Régimen IBEX 35:** {regime} (Precio: {price:.2f} / SMA200: {sma:.2f})",
        f"- **Estrategia Recomendada:** {'COMPRA LENTA (BULL)' if 'ALCISTA' in regime else 'CARRERA RÁPIDA (BEAR)'}\n",
        f"---",
        f"## 🎯 ORDENES OPERATIVAS PARA HOY",
        f"| Ticker | Acción | Inversión (EUR) | Precio Entrada | Stop Loss (4.7%) | Confianza |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for tk in tickers:
        preds, idx = data[f"preds_{tk}"], pd.to_datetime(data[f"idx_{tk}"])
        c, v = data[f"c_{tk}"], data[f"v_{tk}"]
        last_c, last_v = c[-1], v[-1]
        last_pred = preds[-1]
        clase, prob = np.argmax(last_pred), last_pred[np.argmax(last_pred)]
        
        # Filtro de señales
        if clase == 1 and prob >= PROB_UMBRAL:
            tier = get_tier(tk, last_c * last_v)
            inv_eur = CAPITAL_TOTAL * TIER_ALLOC[tier]
            sl_val = last_c * (1 - SL_PCT)
            md.append(f"| **{tk}** | 🔵 COMPRAR | **{inv_eur:,.2f}€** | {last_c:.3f} | {sl_val:.3f} | `{prob*100:.1f}%` |")

    md.append("\n## 📋 Protocolo de Gestión (Reglas TFM)")
    md.append(f"1. **Entrada**: Ejecutar órdenes al precio de apertura de la siguiente vela de 1h.")
    md.append(f"2. **Stop Loss**: Colocar orden de venta limitada al precio indicado (4.7% de riesgo).")
    md.append(f"3. **Salida por Tiempo**: Cerrar la posición tras **48 horas** de mercado si no ha saltado el Stop.")
    md.append(f"4. **Diversificación**: No arriesgar más del 20% del capital total en un solo sector.")

    md.append(f"\n---")
    md.append(f"### 🧪 Health Check de Datos")
    md.append("| Activo | Estado | Último Dato |")
    md.append("| :--- | :--- | :--- |")
    for tk in tickers:
        idx = pd.to_datetime(data[f"idx_{tk}"])
        status = "🟢" if (datetime.now() - idx[-1]).total_seconds()/3600 < 48 else "🔴"
        md.append(f"| {tk} | {status} | {idx[-1].strftime('%Y-%m-%d %H:%M')} |")

    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write("\n".join(md))

if __name__ == "__main__": main()

