import os
import pandas as pd
import numpy as np
import keras
from pathlib import Path

# --- CONFIGURACIÓN ---
os.environ["KERAS_BACKEND"] = "torch"
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
DATA_DIR = BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw"
MODEL_PATH = BASE_DIR / "Modelo_IA_Entrenado.keras"
OUT_DIR = BASE_DIR / "EXPERIMENTO_SP500" / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROB_UMBRAL = 0.93

def build_features(o, h, l, c, v, target_len=60):
    eps = 1e-8
    n = len(c)
    if n < 5: return np.zeros((target_len, 8))
    log_ret = np.zeros(n); log_ret[1:] = np.diff(np.log(np.maximum(c, eps)))
    hl_ratio = (h - l) / np.maximum(c, eps)
    cr = np.maximum(h - l, eps)
    su = (h - np.maximum(o, c)) / cr; sd = (np.minimum(o, c) - l) / cr; br = np.abs(c - o) / cr
    vm = np.mean(v) if np.mean(v) > 0 else eps; vr = np.clip(v / (vm + eps), 0, 10)
    mom5 = np.zeros(n)
    for t in range(5, n): mom5[t] = (c[t] - c[t-5]) / np.maximum(c[t-5], eps)
    cmin, cmax = c.min(), c.max(); cn = (c - cmin) / max(cmax - cmin, eps)
    feat = np.stack([log_ret, hl_ratio, su, sd, br, vr, mom5, cn], axis=1)
    if n >= target_len: feat = feat[-target_len:]
    else: feat = np.vstack([np.zeros((target_len - n, 8)), feat])
    return feat.astype(np.float32)

def run_experiment():
    print(f"Cargando modelo: {MODEL_PATH}")
    model = keras.models.load_model(str(MODEL_PATH))
    spy = pd.read_parquet(DATA_DIR / "SPY_1D.parquet")
    spy['sma200'] = spy['Close'].rolling(200).mean()
    results = []
    files = list(DATA_DIR.glob("*_1H.parquet"))
    print(f"Procesando {len(files)} activos del S&P 500 (Batch Mode)...")
    for f in files:
        ticker = f.name.split("_")[0]
        if ticker == "SPY": continue
        print(f" > {ticker}", end=" ", flush=True)
        df = pd.read_parquet(f)
        if len(df) < 200: print("Skip"); continue
        o, h, l, c, v = [df[col].values.flatten() for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        windows = [build_features(o[i-60:i], h[i-60:i], l[i-60:i], c[i-60:i], v[i-60:i]) for i in range(60, len(df))]
        X = np.array(windows)
        preds = model.predict(X, batch_size=512, verbose=0)
        trades = []
        last_exit = -1
        for i_idx, pred in enumerate(preds):
            i = i_idx + 60
            if i <= last_exit: continue
            p_idx = np.argmax(pred)
            if p_idx == 1 and pred[p_idx] > PROB_UMBRAL:
                fecha = df.index[i]
                try: is_bull = spy.loc[fecha.floor('D'), 'Close'] > spy.loc[fecha.floor('D'), 'sma200']
                except: is_bull = True
                entry_price = c[i]
                if is_bull:
                    exit_i = min(i + 150, len(df) - 1); max_p = entry_price; exit_price = c[exit_i]
                    for j in range(i+1, exit_i+1):
                        if h[j] > max_p: max_p = h[j]
                        if l[j] < max_p * 0.945: exit_i = j; exit_price = max_p * 0.945; break
                else:
                    exit_i = min(i + 48, len(df) - 1); exit_price = c[exit_i]
                trades.append({'ticker': ticker, 'entry_dt': fecha, 'exit_dt': df.index[exit_i], 'gain': (exit_price - entry_price)/entry_price})
                last_exit = exit_i + 3
        if trades:
            ticker_df = pd.DataFrame(trades); results.append(ticker_df)
            print(f"({len(trades)} trades, {ticker_df['gain'].mean():.2%})")
        else: print("(0 trades)")
    if results:
        final_df = pd.concat(results); final_df.to_csv(OUT_DIR / "trades_sp500.csv", index=False)
        print(f"\nTotal trades: {len(final_df)}")

if __name__ == "__main__":
    run_experiment()
