import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# --- CONFIGURACIÓN ---
CACHE_PATH = Path("cache_predicciones_LIVE.npz")
PROB_UMBRAL = 0.90 
REPORT_PATH = Path("README.md")

def get_market_regime():
    try:
        ibex = yf.download("^IBEX", period="1y", progress=False, auto_adjust=True)
        close = ibex["Close"]
        sma200 = close.rolling(window=200).mean().iloc[-1]
        last = close.iloc[-1]
        is_bull = last > sma200
        regime = "🟢 ALCISTA (BULL)" if is_bull else "🔴 BAJISTA/LATERAL (BEAR)"
        return regime, last, sma200
    except:
        return "Desconocido", 0, 0

def main():
    if not CACHE_PATH.exists(): return
    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    regime_str, ibex_price, sma200 = get_market_regime()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = [
        f"# 📈 Dashboard de Trading IA - IBEX 35\n",
        f"**Última actualización:** `{now_str} UTC`\n",
        f"## 🌐 Estado del Mercado",
        f"- **Régimen Actual:** {regime_str}",
        f"- **IBEX 35:** {ibex_price:.2f} (SMA200: {sma200:.2f})\n",
        f"---",
        f"## 🚀 Señales de Inferencia (Top Signals)",
        f"| Ticker | Señal | Confianza | Última Fecha | Estado Datos |",
        f"| :--- | :--- | :--- | :--- | :--- |"
    ]

    found = 0
    for tk in tickers:
        preds, idx = data[f"preds_{tk}"], pd.to_datetime(data[f"idx_{tk}"])
        last_date = idx[-1]
        last_pred = preds[-1]
        clase, prob = np.argmax(last_pred), last_pred[np.argmax(last_pred)]
        quality = "🟢 OK" if (datetime.now() - last_date).total_seconds()/3600 < 48 else "🔴 DESACTUALIZADO"
        
        if clase == 1 and prob >= PROB_UMBRAL:
            found += 1
            md.append(f"| **{tk}** | COMPRA | `{prob*100:.1f}%` | {last_date.strftime('%Y-%m-%d %H:%M')} | {quality} |")

    if found == 0: md.append("| - | *Esperando señales claras...* | - | - | - |")

    md.append("\n## 🛠️ Chequeo de Calidad de Datos (Health Check)")
    md.append("| Ticker | Velas | Último Cierre | Estado |")
    md.append("| :--- | :--- | :--- | :--- |")
    for tk in tickers:
        idx = pd.to_datetime(data[f"idx_{tk}"])
        q = "🟢" if (datetime.now() - idx[-1]).total_seconds()/3600 < 48 else "🔴"
        md.append(f"| {tk} | {len(idx)} | {idx[-1].strftime('%Y-%m-%d')} | {q} |")

    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write("\n".join(md))

if __name__ == "__main__": main()
