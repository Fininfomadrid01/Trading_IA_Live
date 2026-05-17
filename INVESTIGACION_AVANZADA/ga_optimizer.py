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
DATA_DIR = BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw" # Usamos NASDAQ para entrenar el GA

# Parámetros del GA
POBLACION_SIZE = 10
GENERACIONES = 5
MUTACION_RATE = 0.2

def build_features(o, h, l, c, v, target_len=60):
    eps = 1e-8; n = len(c)
    if n < 5: return np.zeros((target_len, 8))
    log_ret = np.zeros(n); log_ret[1:] = np.diff(np.log(np.maximum(c, eps)))
    hl_ratio = (h - l) / np.maximum(c, eps); cr = np.maximum(h - l, eps)
    su = (h-np.maximum(o,c))/cr; sd = (np.minimum(o,c)-l)/cr; br = np.abs(c-o)/cr
    vm = np.mean(v) if np.mean(v) > 0 else eps; vr = np.clip(v/(vm+eps), 0, 10)
    mom5 = np.zeros(n)
    for t in range(5, n): mom5[t] = (c[t]-c[t-5])/np.maximum(c[t-5], eps)
    cmin, cmax = c.min(), c.max(); cn = (c-cmin)/max(cmax-cmin, eps)
    feat = np.stack([log_ret, hl_ratio, su, sd, br, vr, mom5, cn], axis=1)
    if n >= target_len: feat = feat[-target_len:]
    else: feat = np.vstack([np.zeros((target_len - n, 8)), feat])
    return feat.astype(np.float32)

class Individuo:
    def __init__(self, trailing=None, exit_h=None, prob=None):
        self.trailing = trailing if trailing else random.uniform(3.0, 9.0)
        self.exit_h = exit_h if exit_h else random.randint(24, 120)
        self.prob = prob if prob else random.uniform(0.90, 0.96)
        self.fitness = 0.0
        self.pf = 0.0

def evaluate_fitness(individuo, model, tickers_data):
    all_gains = []
    for ticker, df in tickers_data.items():
        o, h, l, c, v = [df[col].values.flatten() for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        # Simplificamos: solo analizamos los últimos 500 barras para velocidad del GA
        start_i = max(60, len(df) - 500)
        windows = [build_features(o[i-60:i], h[i-60:i], l[i-60:i], c[i-60:i], v[i-60:i]) for i in range(start_i, len(df))]
        preds = model.predict(np.array(windows), batch_size=512, verbose=0)
        
        last_exit = -1
        for idx, pred in enumerate(preds):
            i = idx + start_i
            if i <= last_exit: continue
            if np.argmax(pred) == 1 and pred[1] > individuo.prob:
                entry_p = c[i]
                exit_i = min(i + individuo.exit_h, len(df)-1)
                max_p = entry_p; exit_p = c[exit_i]
                for j in range(i+1, exit_i+1):
                    if h[j] > max_p: max_p = h[j]
                    if l[j] < max_p * (1 - individuo.trailing/100):
                        exit_i = j; exit_p = max_p * (1 - individuo.trailing/100); break
                all_gains.append((exit_p - entry_p)/entry_p)
                last_exit = exit_i + 3
                
    if not all_gains: return 0.0, 0.0
    gains = [g for g in all_gains if g > 0]
    losses = [abs(g) for g in all_gains if g < 0]
    pf = sum(gains) / sum(losses) if losses else 2.0
    # Fitness = PF + log(n_trades) para evitar PF altos con 1 trade
    fitness = pf * (1 + np.log10(len(all_gains)))
    return fitness, pf

def run_ga():
    print("Iniciando Optimización Genética...")
    model = keras.models.load_model(str(MODEL_PATH))
    # Cargamos 5 tickers para velocidad de evolución
    tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]
    data = {t: pd.read_parquet(DATA_DIR / f"{t}_1H.parquet") for t in tickers}
    
    poblacion = [Individuo() for _ in range(POBLACION_SIZE)]
    
    for gen in range(GENERACIONES):
        print(f"\n--- Generación {gen+1}/{GENERACIONES} ---")
        for ind in poblacion:
            ind.fitness, ind.pf = evaluate_fitness(ind, model, data)
            print(f"Individuo: Trail={ind.trailing:.1f}%, Exit={ind.exit_h}h, Prob={ind.prob:.3f} -> PF: {ind.pf:.2f}")
        
        poblacion.sort(key=lambda x: x.fitness, reverse=True)
        # Selección (Elites)
        padres = poblacion[:4]
        descendencia = []
        while len(descendencia) < POBLACION_SIZE - 4:
            p1, p2 = random.sample(padres, 2)
            # Crossover
            hijo = Individuo(
                trailing=(p1.trailing + p2.trailing)/2,
                exit_h=int((p1.exit_h + p2.exit_h)/2),
                prob=(p1.prob + p2.prob)/2
            )
            # Mutación
            if random.random() < MUTACION_RATE: hijo.trailing += random.uniform(-1, 1)
            if random.random() < MUTACION_RATE: hijo.exit_h += random.randint(-10, 10)
            descendencia.append(hijo)
        
        poblacion = padres + descendencia

    best = poblacion[0]
    print(f"\nMEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"Trailing Stop: {best.trailing:.2f}%")
    print(f"Exit Hours:    {best.exit_h}")
    print(f"Prob Threshold: {best.prob:.4f}")
    print(f"Profit Factor:  {best.pf:.2f}")

if __name__ == "__main__":
    run_ga()
