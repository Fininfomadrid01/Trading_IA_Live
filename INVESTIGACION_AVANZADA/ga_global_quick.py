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
    feat = np.stack([log_ret, (h-l)/c, (h-c)/c, (c-l)/c, (c-o)/c, v/np.mean(v), (c-c)/c, (c-c)/c], axis=1) # Simplified for speed in GA
    if n >= target_len: feat = feat[-target_len:]
    else: feat = np.vstack([np.zeros((target_len-n, 8)), feat])
    return feat.astype(np.float32)

class Individuo:
    def __init__(self, trailing=None, exit_h=None, prob=None):
        self.trailing = trailing if trailing else random.uniform(3.0, 9.0)
        self.exit_h = exit_h if exit_h else random.randint(24, 120)
        self.prob = prob if prob else random.uniform(0.90, 0.96)
        self.fitness = 0.0; self.pf = 0.0

def evaluate_ga(mercado, data_dir, tickers):
    print(f"\n>>> Evolucionando {mercado}...")
    model = keras.models.load_model(str(MODEL_PATH))
    data = {t: pd.read_parquet(data_dir / f"{t}_1H.parquet") for t in tickers}
    
    poblacion = [Individuo() for _ in range(8)]
    for gen in range(3): # 3 generaciones rápidas para estimar
        for ind in poblacion:
            all_gains = []
            for t, df in data.items():
                c = df['Close'].values.flatten()
                h = df['High'].values.flatten()
                l = df['Low'].values.flatten()
                start_i = max(60, len(df)-500)
                # Inferencia rápida
                for i in range(start_i, len(df), 5): # Saltamos de 5 en 5 para el GA rápido
                    # Simulamos señal aleatoria basada en prob para estimar el impacto del stop/time
                    if random.random() < 0.1: 
                        entry_p = c[i]
                        exit_i = min(i + ind.exit_h, len(df)-1)
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
                ind.pf = sum(gains)/sum(losses) if losses else 2.0
            ind.fitness = ind.pf
        poblacion.sort(key=lambda x: x.fitness, reverse=True)
    
    best = poblacion[0]
    print(f"Mejor {mercado}: Trail={best.trailing:.2f}%, Exit={best.exit_h}h, PF_Est={best.pf:.2f}")
    return best

if __name__ == "__main__":
    # FTSE
    ftse_best = evaluate_ga("FTSE 100", BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw", ["SHEL.L", "AZN.L", "HSBA.L"])
    # S&P
    sp_best = evaluate_ga("S&P 500", BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw", ["AAPL", "MSFT", "GOOGL"])
