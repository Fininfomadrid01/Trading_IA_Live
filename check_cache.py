import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
cache_path = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"

if cache_path.exists():
    data = np.load(str(cache_path))
    keys = list(data.keys())
    print("Number of keys in cache:", len(keys))
    # Find tickers
    tickers = sorted(list(set([k.split('_')[1] for k in keys if k.startswith('preds_')])))
    print("Tickers:", tickers)
    print("Number of tickers:", len(tickers))
    
    # Print info for first ticker
    if tickers:
        tk = tickers[0]
        preds = data[f"preds_{tk}"]
        idx = pd.DatetimeIndex(data[f"idx_{tk}"])
        c = data[f"c_{tk}"]
        print(f"\nTicker: {tk}")
        print("preds shape:", preds.shape)
        print("index shape:", idx.shape)
        print("close shape:", c.shape)
        print("First date:", idx[0])
        print("Last date:", idx[-1])
else:
    print("Cache not found at:", cache_path)
