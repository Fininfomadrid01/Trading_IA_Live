import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS")
cache_path = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"

if cache_path.exists():
    data = np.load(str(cache_path))
    keys = list(data.keys())
    tickers = sorted(list(set([k.split('_')[1] for k in keys if k.startswith('preds_')])))
    print(f"Tickers en CACHE_BASE ({len(tickers)}):")
    print(tickers)
else:
    print("No se encuentra la cache base.")
