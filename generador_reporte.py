import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# --- CONFIGURACIÓN MAESTRA ---
CACHE_PATH = Path("cache_predicciones_LIVE.npz")
REPORT_PATH = Path("README.md")

MARKET_CONFIGS = {
    'IBEX':   {'suffix': '.MC', 'prob': 0.923, 'emoji': '🇪🇸'},
    'FTSE':   {'suffix': '.L',  'prob': 0.935, 'emoji': '🇬🇧'},
    'USA':    {'suffix': '',    'prob': 0.954, 'emoji': '🇺🇸'},
    'CRYPTO': {'suffix': '-USD', 'prob': 0.923, 'emoji': '₿'},
    'COMMO':  {'suffix': '=F',   'prob': 0.933, 'emoji': '🏆'} # Oro/Petróleo
}

def get_market_regimes():
    regimes = {}
    # Sincronizamos los 4 pilares
    indices = {
        "IBEX 35": "^IBEX", 
        "NASDAQ 100": "^NDX", 
        "Bitcoin": "BTC-USD", 
        "ORO (Gold)": "GC=F"
    }
    for name, ticker in indices.items():
        try:
            data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
            if data.empty: continue
            close = data["Close"]
            sma200 = close.rolling(window=200).mean().iloc[-1]
            last = close.iloc[-1]
            is_bull = last > sma200
            regStr = "🟢 BULL" if is_bull else "🔴 BEAR"
            regimes[name] = (regStr, last)
        except: pass
    return regimes

def get_config_for_ticker(tk):
    if tk.endswith(".MC"): return MARKET_CONFIGS['IBEX']
    if tk.endswith(".L"):  return MARKET_CONFIGS['FTSE']
    if tk.endswith("-USD"): return MARKET_CONFIGS['CRYPTO']
    if tk.endswith("=F") or tk in ["GOLD", "OIL"]: return MARKET_CONFIGS['COMMO']
    return MARKET_CONFIGS['USA']

def main():
    if not CACHE_PATH.exists():
        print("ERROR: No hay cache.")
        return

    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    regimes = get_market_regimes()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- RENDERIZADO DEL DASHBOARD DE HONOR ---
    md = [
        f"# 🏆 TFM: Sistema de Inferencia Multiactivo Global",
        f"**Estado General:** `SISTEMA OPTIMIZADO` | **Última sincronización:** `{now_str} UTC`\n",
        f"Este dashboard presenta las inferencias en tiempo real de la arquitectura CNN-BiLSTM-Attention validada en mercados globales de renta variable, criptoactivos y materias primas.\n",
        f"## 🌐 Monitor de Regímenes Globales",
        f"| Mercado | Estado | Último Precio |",
        f"| :--- | :--- | :--- |"
    ]
    
    for name, (regStr, price) in regimes.items():
        md.append(f"| {name} | {regStr} | {price:.2f} |")

    md.append("\n---\n## 🚀 Señales de Inferencia (Top Opportunities)")
    md.append(f"| Mercado | Ticker | Señal | Confianza | Salud de Datos |")
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
            
            time_diff = (datetime.now() - last_date).total_seconds() / 3600
            quality_emoji = "🟢 OK" if time_diff < 48 else "🔴"
            
            if clase == 1 and prob >= cfg['prob']:
                signals_found += 1
                md.append(f"| {cfg['emoji']} | **{tk}** | COMPRA | `{prob*100:.1f}%` | {quality_emoji} |")
        except: pass

    if signals_found == 0:
        md.append("| - | - | *Esperando configuraciones de alta probabilidad...* | - | - |")

    md.append("\n## 📊 Resumen de Tesis (Milestones)")
    md.append("- ✅ **IBEX 35**: Validación histórica 2019-2025.")
    md.append("- ✅ **NASDAQ/S&P**: Generalización internacional lograda.")
    md.append("- ✅ **CRYPTO**: Profit Factor de 1.60 detectado.")
    md.append("- ✅ **COMMODITIES**: Profit Factor de 3.19 detectado en Oro/Petróleo.")
    
    md.append("\n---\n*Dashboard automatizado para la entrega final del Trabajo Fin de Máster (2026).*")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"¡Dashboard de Honor Global generado exitosamente en {REPORT_PATH}!")

if __name__ == "__main__":
    main()
