import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"

PROB_UMBRAL = 0.93
SL_PCT = 0.047
HORIZONTE = 48

def analyze_by_market():
    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    
    results = {}
    
    # Categorización simple
    for tk in tickers:
        market = "OTROS"
        if tk in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']: market = "CRYPTO"
        elif tk in ['GOLD', 'OIL', 'SILVER', 'GAS']: market = "COMMODITIES"
        elif tk in ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NFLX', 'AMD', 'AVGO', 'QQQ']: market = "USA"
        elif tk in ['ACS', 'ALM', 'ANA', 'AMS', 'BBVA', 'BKT', 'CABK', 'CLNX', 'COL', 'ENG', 'ELE', 'FER', 'FLUID', 'GRF', 'IAG', 'IBE', 'ITX', 'IDR', 'MAP', 'MEL', 'MRL', 'NTGY', 'REE', 'REP', 'ROVI', 'SAB', 'SAN', 'SCYR', 'SLBA', 'TEF', 'TRE', 'UNI', 'CIE', 'DOM', 'LDA', 'PHM', 'SLR', 'VIS']: market = "IBEX/MC"
        
        if market not in results: results[market] = []
        
        p = data[f"preds_{tk}"]
        c = data[f"c_{tk}"]
        l = data[f"l_{tk}"]
        
        for i in range(60, len(p)):
            if p[i][1] >= PROB_UMBRAL:
                entry_p = c[i]
                sl = entry_p * (1 - SL_PCT)
                exit_i = min(i + HORIZONTE, len(c) - 1)
                res = "TIME"
                for j in range(i+1, exit_i+1):
                    if l[j] <= sl: exit_i = j; res = "SL"; break
                exit_p = sl if res == "SL" else c[exit_i]
                results[market].append((exit_p - entry_p) / entry_p)

    print(f"{'MERCADO':<15} | {'TRADES':<8} | {'WIN RATE':<10} | {'EV%':<10}")
    print("-" * 50)
    for m, trades in results.items():
        if not trades: continue
        wr = len([t for t in trades if t > 0]) / len(trades)
        ev = np.mean(trades)
        print(f"{m:<15} | {len(trades):<8} | {wr*100:>8.2f}% | {ev*100:>+8.3f}%")

if __name__ == "__main__":
    analyze_by_market()
