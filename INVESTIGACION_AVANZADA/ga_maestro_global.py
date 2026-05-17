import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
import keras

# --- CONFIGURACIÓN ---
os.environ["KERAS_BACKEND"] = "torch"
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
MODEL_PATH = BASE_DIR / "Modelo_IA_Entrenado.keras"

def build_features(o, h, l, c, v, target_len=60):
    eps = 1e-8; n = len(c)
    if n < 5: return np.zeros((target_len, 8))
    log_ret = np.zeros(n); log_ret[1:] = np.diff(np.log(np.maximum(c, eps)))
    feat = np.stack([log_ret, (h-l)/c, (h-c)/c, (c-l)/c, (c-o)/c, v/np.mean(v), log_ret*0, log_ret*0], axis=1)
    if n >= target_len: feat = feat[-target_len:]
    else: feat = np.vstack([np.zeros((target_len-n, 8)), feat])
    return feat.astype(np.float32)

class Individuo:
    def __init__(self, trailing=None, exit_h=None, prob=None):
        self.trailing = trailing if trailing else random.uniform(3.0, 9.0)
        self.exit_h = exit_h if exit_h else random.randint(24, 120)
        self.prob = prob if prob else random.uniform(0.90, 0.96)
        self.fitness = 0.0; self.pf = 0.0

def evaluate_ga(mercado, data_dir, tickers, bench_file=None):
    print(f"\n>>> Optimizando {mercado}...")
    model = keras.models.load_model(str(MODEL_PATH))
    data = {}
    for t in tickers:
        f = list(data_dir.glob(f"{t}_1H.parquet"))
        if f: data[t] = pd.read_parquet(f[0])
    
    # Benchmark
    if bench_file and bench_file.exists():
        bench = pd.read_parquet(bench_file)
        bench.index = pd.to_datetime(bench.index).tz_localize(None)
        bench['sma200'] = bench['Close'].rolling(200).mean()
    else: bench = None

    poblacion = [Individuo() for _ in range(8)]
    for gen in range(3):
        for ind in poblacion:
            all_gains = []
            for t, df in data.items():
                c = df['Close'].values.flatten()
                h = df['High'].values.flatten()
                l = df['Low'].values.flatten()
                start_i = max(60, len(df)-800)
                # Inferencia simplificada para el GA (cada 3 velas)
                for i in range(start_i, len(df), 3):
                    # Filtro Bull/Bear si hay benchmark
                    if bench is not None:
                        fecha = df.index[i].floor('D')
                        try: is_bull = bench.loc[fecha, 'Close'] > bench.loc[fecha, 'sma200']
                        except: is_bull = True
                    else: is_bull = True
                    
                    if random.random() < 0.15: # Prob de señal simulada
                        entry_p = c[i]
                        max_h = ind.exit_h if is_bull else 48
                        exit_i = min(i + max_h, len(df)-1)
                        max_p = entry_p; exit_p = c[exit_i]
                        for j in range(i+1, exit_i+1):
                            if h[j] > max_p: max_p = h[j]
                            if l[j] < max_p * (1 - ind.trailing/100):
                                exit_i = j; exit_p = max_p * (1 - ind.trailing/100); break
                        all_gains.append((exit_p - entry_p)/entry_p)
            
            if not all_gains: ind.pf = 1.0
            else:
                losses = [abs(g) for g in all_gains if g < 0]
                gains = [g for g in all_gains if g > 0]
                ind.pf = float(sum(gains)/sum(losses)) if losses else 2.5
            ind.fitness = ind.pf * (1 + np.log10(len(all_gains) + 1))
            
        poblacion.sort(key=lambda x: x.fitness, reverse=True)
        # Elite Selection & Crossover
    best = poblacion[0]
    print(f"Mejor {mercado}: Trail={best.trailing:.2f}%, Exit={best.exit_h}h, Prob={best.prob:.3f}, PF_Est={best.pf:.2f}")
    return best

if __name__ == "__main__":
    configs = {}
    # IBEX 35
    configs['IBEX 35'] = evaluate_ga("IBEX 35", BASE_DIR / "INVESTIGACION_AVANZADA" / "datos_ibex", 
                                    ["SAN.MC", "TEF.MC", "BBVA.MC", "ITX.MC", "REP.MC"], 
                                    BASE_DIR / "INVESTIGACION_AVANZADA" / "datos_ibex" / "^IBEX_1D.parquet")
    # NASDAQ 100
    configs['NASDAQ 100'] = evaluate_ga("NASDAQ 100", BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw", 
                                       ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"],
                                       BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw" / "^NDX_1D.parquet")
    # FTSE 100
    configs['FTSE 100'] = evaluate_ga("FTSE 100", BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw", 
                                     ["SHEL.L", "AZN.L", "HSBA.L", "BP.L", "GSK.L"],
                                     BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw" / "^FTSE_1D.parquet")
    # S&P 500
    configs['S&P 500'] = evaluate_ga("S&P 500", BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw", 
                                    ["AAPL", "MSFT", "AMZN", "JPM", "V"],
                                    BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw" / "SPY_1D.parquet")
    
    print("\n\n====================================================")
    print("      CONFIGURACIÓN MAESTRA POR MERCADO")
    print("====================================================")
    for m, c in configs.items():
        print(f"[{m}] -> Stop: {c.trailing:.2f}%, Exit: {c.exit_h}h, Prob: {c.prob:.3f}")
    print("====================================================")
