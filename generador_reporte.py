import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# --- CONFIGURACIÓN ---
CACHE_PATH = Path("cache_predicciones_LIVE.npz")
REPORT_PATH = Path("README.md")

# Umbrales optimizados por mercado
MARKET_CONFIGS = {
    'IBEX':   {'suffix': '.MC', 'prob': 0.923, 'emoji': '🇪🇸'},
    'FTSE':   {'suffix': '.L',  'prob': 0.935, 'emoji': '🇬🇧'},
    'USA':    {'suffix': '',    'prob': 0.954, 'emoji': '🇺🇸'}
}

def get_market_regimes():
    regimes = {}
    indices = {"IBEX 35": "^IBEX", "NASDAQ 100": "^NDX", "S&P 500": "SPY"}
    for name, ticker in indices.items():
        try:
            data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
            if data.empty: 
                regimes[name] = ("Desconocido", 0)
                continue
            close = data["Close"]
            sma200 = close.rolling(window=200).mean().iloc[-1]
            last = close.iloc[-1]
            is_bull = last > sma200
            regStr = "🟢 BULL" if is_bull else "🔴 BEAR"
            regimes[name] = (regStr, last)
        except:
            regimes[name] = ("Error", 0)
    return regimes

def get_config_for_ticker(tk):
    if tk.endswith(".MC"): return MARKET_CONFIGS['IBEX']
    if tk.endswith(".L"):  return MARKET_CONFIGS['FTSE']
    return MARKET_CONFIGS['USA']

def main():
    if not CACHE_PATH.exists():
        print("ERROR: No hay cache.")
        return

    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    
    regimes = get_market_regimes()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- INICIO DEL DASHBOARD GLOBAL ---
    md = [
        f"# 🌍 Dashboard Global de Trading IA",
        f"**Estado del Sistema:** `OPERATIVO` | **Última sincronización:** `{now_str} UTC`\n",
        f"## 🌐 Monitor de Mercados (Regime Switching)",
        f"| Mercado | Estado Actual | Último Cierre |",
        f"| :--- | :--- | :--- |"
    ]
    
    for name, (regStr, price) in regimes.items():
        md.append(f"| {name} | {regStr} | {price:.2f} |")

    md.append("\n---\n## 🚀 Señales de Inferencia (Optimización Genética)")
    md.append(f"| Mercado | Ticker | Señal | Confianza | Salud Datos |")
    md.append(f"| :--- | :--- | :--- | :--- | :--- |")

    signals_found = 0
    for tk in tickers:
        try:
            preds = data[f"preds_{tk}"]
            idx = pd.to_datetime(data[f"idx_{tk}"])
            last_date = idx[-1]
            
            cfg = get_config_for_ticker(tk)
            last_pred = preds[-1]
            clase = np.argmax(last_pred)
            prob = last_pred[clase]
            
            # Calidad de datos
            time_diff = (datetime.now() - last_date).total_seconds() / 3600
            quality_emoji = "🟢 OK" if time_diff < 48 else "🔴"
            
            # Filtro por umbral optimizado por mercado
            if clase == 1 and prob >= cfg['prob']:
                signals_found += 1
                md.append(f"| {cfg['emoji']} | **{tk}** | COMPRA | `{prob*100:.1f}%` | {quality_emoji} |")
        except: pass

    if signals_found == 0:
        md.append("| - | - | *Sin señales que superen los umbrales de confianza hoy* | - | - |")

    md.append("\n## 🛠️ Health Check de Activos")
    md.append("| Ticker | Velas en Cache | Último Dato | Estado |")
    md.append("| :--- | :--- | :--- | :--- |")

    for tk in tickers[:20]: # Mostramos los 20 primeros para no saturar el README
        idx = pd.to_datetime(data[f"idx_{tk}"])
        last_date = idx[-1]
        count = len(idx)
        time_diff = (datetime.now() - last_date).total_seconds() / 3600
        quality_emoji = "🟢" if time_diff < 48 else "🔴"
        md.append(f"| {tk} | {count} | {last_date.strftime('%Y-%m-%d')} | {quality_emoji} |")

    md.append("\n---\n*Este Dashboard es el resultado de la investigación 'Generalización de Modelos de Deep Learning en Mercados Globales' (TFM 2026).*")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Dashboard global generado en {REPORT_PATH}")

if __name__ == "__main__":
    main()
