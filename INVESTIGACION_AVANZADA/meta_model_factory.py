import pandas as pd
import numpy as np
import keras
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- CONFIGURACIÓN ---
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
MODEL_PATH = BASE_DIR / "Modelo_IA_Entrenado.keras"
DATA_DIR = BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw"

def build_features(o, h, l, c, v, target_len=60):
    eps = 1e-8; n = len(c)
    if n < 5: return np.zeros((target_len, 8))
    log_ret = np.zeros(n); log_ret[1:] = np.diff(np.log(np.maximum(c, eps)))
    hl_ratio = (h - l) / np.maximum(c, eps); cr = np.maximum(h - l, eps)
    su = (h-np.maximum(o,c))/cr; sd = (np.minimum(o,c)-l)/cr; br = np.abs(c-o)/cr
    vm = np.mean(v) if np.mean(v) > 0 else eps; vr = np.clip(v/(vm+eps), 0, 10)
    mom5 = np.zeros(n)
    for t in range(5, n): mom5[t] = (c[t]-c[t-5]) / np.maximum(c[t-5], eps)
    cmin, cmax = c.min(), c.max(); cn = (c-cmin) / max(cmax - cmin, eps)
    feat = np.stack([log_ret, hl_ratio, su, sd, br, vr, mom5, cn], axis=1)
    if n >= target_len: feat = feat[-target_len:]
    else: feat = np.vstack([np.zeros((target_len - n, 8)), feat])
    return feat.astype(np.float32)

def generate_training_data():
    print("Generando datos de entrenamiento para el Meta-Modelo...")
    model = keras.models.load_model(str(MODEL_PATH))
    tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "NFLX"]
    dataset = []

    for t in tickers:
        print(f"  Procesando {t}...")
        df = pd.read_parquet(DATA_DIR / f"{t}_1H.parquet")
        o, h, l, c, v = [df[col].values.flatten() for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Inferencia por bloques
        windows = [build_features(o[i-60:i], h[i-60:i], l[i-60:i], c[i-60:i], v[i-60:i]) for i in range(60, len(df))]
        preds = model.predict(np.array(windows), batch_size=512, verbose=0)
        
        last_exit = -1
        for idx, pred in enumerate(preds):
            i = idx + 60
            if i <= last_exit: continue
            
            p_idx = np.argmax(pred)
            prob = pred[p_idx]
            
            # Si hay señal alcista (clase 1)
            if p_idx == 1 and prob > 0.90:
                entry_p = c[i]
                exit_i = min(i + 48, len(df)-1)
                gain = (c[exit_i] - entry_p) / entry_p
                
                # Feature Vector para el Meta-Modelo:
                # 1. Probabilidad del modelo principal
                # 2. Volatilidad reciente (ATR-like)
                volatilidad = (h[i-5:i].max() - l[i-5:i].min()) / entry_p
                # 3. RSI-like (momentum)
                mom = (c[i] - c[i-10]) / c[i-10]
                
                dataset.append({
                    'prob_main': prob,
                    'volatilidad': volatilidad,
                    'momentum': mom,
                    'target': 1 if gain > 0.005 else 0 # 1 si gana más de 0.5%, 0 si no
                })
                last_exit = exit_i + 3
                
    return pd.DataFrame(dataset)

def train_meta_model():
    df = generate_training_data()
    print(f"\nTotal ejemplos recolectados: {len(df)}")
    print(f"Clases: {df['target'].value_counts()}")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print("\nREPORTE DE CLASIFICACIÓN DEL META-MODELO:")
    print(classification_report(y_test, preds))
    
    # Importancia de variables
    importancias = pd.Series(clf.feature_importances_, index=X.columns)
    print("\nIMPORTANCIA DE VARIABLES:")
    print(importancias.sort_values(ascending=False))

if __name__ == "__main__":
    train_meta_model()
