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
DATA_DIR = BASE_DIR / "EXPERIMENTO_CRYPTO" / "datos_raw"

class Individuo:
    def __init__(self, trailing=None, exit_h=None, prob=None):
        self.trailing = trailing if trailing else random.uniform(8.0, 18.0) # Más amplio para Crypto
        self.exit_h = exit_h if exit_h else random.randint(12, 72)     # Más rápido para Crypto
        self.prob = prob if prob else random.uniform(0.92, 0.98)        # Más exigente
        self.fitness = 0.0; self.pf = 0.0

def evaluate_ga():
    print(f"\n>>> Optimizando Mercado CRYPTO (BTC, ETH, SOL, BNB, XRP)...")
    model = keras.models.load_model(str(MODEL_PATH))
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    data = {}
    for t in tickers:
        f = DATA_DIR / f"{t}_1H.parquet"
        if f.exists(): data[t] = pd.read_parquet(f)
    
    # Benchmark (Bitcoin como proxy de mercado)
    bench = pd.read_parquet(DATA_DIR / "BTC-USD_1D.parquet")
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    bench['sma200'] = bench['Close'].rolling(200).mean()

    poblacion = [Individuo() for _ in range(10)]
    for gen in range(3):
        for ind in poblacion:
            all_gains = []
            for t, df in data.items():
                c = df['Close'].values.flatten()
                h = df['High'].values.flatten()
                l = df['Low'].values.flatten()
                start_i = max(60, len(df)-2000)
                for i in range(start_i, len(df), 4): # Paso de 4 para velocidad
                    fecha = df.index[i].floor('D')
                    try: is_bull = bench.loc[fecha, 'Close'] > bench.loc[fecha, 'sma200']
                    except: is_bull = True
                    
                    if random.random() < 0.10: # Simulación de señal
                        entry_p = c[i]
                        max_h = ind.exit_h if is_bull else 24
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
        
    best = poblacion[0]
    print(f"RESULTADO CRIPTO: Trail={best.trailing:.2f}%, Exit={best.exit_h}h, Prob={best.prob:.3f}, PF_Est={best.pf:.2f}")
    return best

if __name__ == "__main__":
    evaluate_ga()
