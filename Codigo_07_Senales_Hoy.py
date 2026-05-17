import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# --- CONFIGURACIÓN ---
ROOT = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS")
CACHE_PATH = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"
PROB_UMBRAL = 0.93

def main():
    print("\n" + "="*50)
    print("   SISTEMA DE SEÑALES IBEX 35 - IA REGIME SWITCHING")
    print("="*50)

    # 1. Determinar Régimen de Mercado (IBEX vs SMA200)
    print("Analizando régimen de mercado...")
    ibex = yf.download("^IBEX", period="1y", progress=False)
    if isinstance(ibex.columns, pd.MultiIndex): ibex_close = ibex['Close']['^IBEX']
    else: ibex_close = ibex['Close']
    
    sma200 = ibex_close.rolling(window=200).mean().iloc[-1]
    last_price = ibex_close.iloc[-1]
    is_bull = last_price > sma200
    
    regime_str = "ALCISTA (BULL)" if is_bull else "BAJISTA/LATERAL (BEAR)"
    print(f"  > IBEX 35: {last_price:.2f} | SMA200: {sma200:.2f}")
    print(f"  > RÉGIMEN ACTUAL: {regime_str}")
    print("-" * 50)

    # 2. Leer Predicciones de la IA
    if not CACHE_PATH.exists():
        print("ERROR: Ejecuta primero el Actualizador_Yahoo para tener datos de hoy.")
        return

    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    
    signals = []
    for tk in tickers:
        preds = data[f"preds_{tk}"]
        idx   = pd.to_datetime(data[f"idx_{tk}"])
        last_idx = idx[-1]
        
        # Solo miramos la última vela disponible en la caché
        last_pred = preds[-1]
        clase = np.argmax(last_pred)
        prob  = last_pred[clase]
        
        # Clase 1 es 'ALC_CONT' (Alcista Continuación) en nuestro modelo
        if clase == 1 and prob >= PROB_UMBRAL:
            signals.append({
                "Ticker": tk,
                "Probabilidad": f"{prob*100:.1f}%",
                "Fecha_Vela": last_idx.strftime("%Y-%m-%d %H:%M")
            })

    # 3. Mostrar Resultados
    if not signals:
        print("HOY NO HAY SEÑALES CLARAS. IA en espera.")
    else:
        print(f"¡SEÑALES DETECTADAS PARA MAÑANA ({signals[0]['Fecha_Vela']}):")
        df_sigs = pd.DataFrame(signals)
        print(df_sigs.to_string(index=False))
        print("\nRECOMENDACIÓN DE OPERATIVA:")
        if is_bull:
            print("  > ESTRATEGIA BULL: Compra con Trailing Stop del 5.5%. Objetivo: Largo plazo.")
        else:
            print("  > ESTRATEGIA BEAR: Compra rápida. Cierre fijo a las 48h o Stop Loss del 4.7%.")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
