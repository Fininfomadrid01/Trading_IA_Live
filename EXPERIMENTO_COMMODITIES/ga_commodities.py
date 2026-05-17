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
DATA_DIR = BASE_DIR / "EXPERIMENTO_COMMODITIES" / "datos_raw"

class Individuo:
    def __init__(self, trailing=None, exit_h=None, prob=None):
        self.trailing = trailing if trailing else random.uniform(2.5, 6.5) # Más ajustado para Materias Primas
        self.exit_h = exit_h if exit_h else random.randint(48, 120)    # Maduración más lenta
        self.prob = prob if prob else random.uniform(0.90, 0.95)
        self.fitness = 0.0; self.pf = 0.0

def evaluate_ga():
    print(f"\n>>> Optimizando Mercado COMMODITIES (Oro y Petróleo)...")
    model = keras.models.load_model(str(MODEL_PATH))
    data = {
        "GOLD": pd.read_parquet(DATA_DIR / "GOLD_1H.parquet"),
        "OIL": pd.read_parquet(DATA_DIR / "OIL_1H.parquet")
    }
    
    poblacion = [Individuo() for _ in range(10)]
    for gen in range(3):
        for ind in poblacion:
            all_gains = []
            for name, df in data.items():
                c = df['Close'].values.flatten()
                h = df['High'].values.flatten()
                l = df['Low'].values.flatten()
                start_i = max(60, len(df)-2500)
                for i in range(start_i, len(df), 4):
                    if random.random() < 0.12: # Señal simulada
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
                ind.pf = float(sum(gains)/sum(losses)) if losses else 2.5
            ind.fitness = ind.pf * (1 + np.log10(len(all_gains) + 1))
            
        poblacion.sort(key=lambda x: x.fitness, reverse=True)
        
    best = poblacion[0]
    print(f"RESULTADO COMMODITIES: Trail={best.trailing:.2f}%, Exit={best.exit_h}h, Prob={best.prob:.3f}, PF_Est={best.pf:.2f}")
    return best

if __name__ == "__main__":
    evaluate_ga()
