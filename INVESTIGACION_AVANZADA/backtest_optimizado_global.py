import os
import pandas as pd
import numpy as np
import keras
from pathlib import Path

# --- CONFIGURACIÓN ---
os.environ["KERAS_BACKEND"] = "torch"
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
MODEL_PATH = BASE_DIR / "Modelo_IA_Entrenado.keras"

# CONFIGURACIONES MAESTRAS
CONFIGS = {
    'IBEX 35':    {'stop': 5.65, 'exit': 81,  'prob': 0.923, 'dir': BASE_DIR / "INVESTIGACION_AVANZADA" / "datos_ibex", 'bench': "^IBEX_1D.parquet"},
    'NASDAQ 100': {'stop': 6.19, 'exit': 87,  'prob': 0.954, 'dir': BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw",     'bench': "QQQ_1D.parquet"},
    'FTSE 100':   {'stop': 3.92, 'exit': 110, 'prob': 0.935, 'dir': BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw",    'bench': "^FTSE_1D.parquet"},
    'S&P 500':    {'stop': 6.67, 'exit': 115, 'prob': 0.901, 'dir': BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw",      'bench': "SPY_1D.parquet"}
}

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

def run_optimized_backtest():
    print("Iniciando Backtesting Optimizado Global...")
    model = keras.models.load_model(str(MODEL_PATH))
    final_report = []

    for mercado, cfg in CONFIGS.items():
        print(f"\n--- Analizando {mercado} con Setup Maestro ---")
        data_dir = cfg['dir']
        tickers_files = list(data_dir.glob("*_1H.parquet"))[:15] # 15 activos por mercado para muestra robusta
        
        # Benchmark
        bench_df = pd.read_parquet(data_dir / cfg['bench'])
        bench_df.index = pd.to_datetime(bench_df.index).tz_localize(None)
        bench_df['sma200'] = bench_df['Close'].rolling(200).mean()
        
        all_trades = []
        print(f"  Encontrados {len(tickers_files)} archivos de datos.")
        for f in tickers_files:
            if f.name.startswith("^") or f.name.startswith("SPY") or f.name.startswith("QQQ"): continue
            df = pd.read_parquet(f)
            o, h, l, c, v = [df[col].values.flatten() for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            start_i = max(60, len(df)-1200)
            windows = [build_features(o[i-60:i], h[i-60:i], l[i-60:i], c[i-60:i], v[i-60:i]) for i in range(start_i, len(df))]
            if not windows: continue
            
            preds = model.predict(np.array(windows), batch_size=512, verbose=0)
            max_probs = preds[:, 1]
            n_signals = np.sum(max_probs > cfg['prob'])
            if n_signals > 0:
                print(f"    {f.name}: {n_signals} señales potenciales detectadas.")
            
            last_exit = -1
            for idx, pred in enumerate(preds):
                i = idx + start_i
                if i <= last_exit: continue
                if np.argmax(pred) == 1 and pred[1] > cfg['prob']:
                    fecha = df.index[i].floor('D')
                    try: is_bull = bench_df.loc[fecha, 'Close'] > bench_df.loc[fecha, 'sma200']
                    except: is_bull = True
                    
                    entry_p = c[i]
                    max_h_time = cfg['exit'] if is_bull else 48
                    exit_i = min(i + max_h_time, len(df)-1)
                    max_p = entry_p; exit_p = c[exit_i]
                    for j in range(i+1, exit_i+1):
                        if h[j] > max_p: max_p = h[j]
                        if l[j] < max_p * (1 - cfg['stop']/100):
                            exit_i = j; exit_p = max_p * (1 - cfg['stop']/100); break
                    all_trades.append((exit_p - entry_p)/entry_p)
                    last_exit = exit_i + 3
        
        if all_trades:
            gains = [g for g in all_trades if g > 0]
            losses = [abs(g) for g in all_trades if g < 0]
            pf = sum(gains)/sum(losses) if losses else 2.5
            wr = len(gains)/len(all_trades)
            avg = np.mean(all_trades)
            final_report.append({'Mercado': mercado, 'PF': pf, 'WR': wr, 'Avg': avg, 'Trades': len(all_trades)})
            print(f"RESULTADO {mercado}: PF={pf:.2f}, WR={wr:.1%}, Ganancia Media={avg:.2%}")

    print("\n\n====================================================")
    print("      RESUMEN DE RENDIMIENTO OPTIMIZADO")
    print("====================================================")
    report_df = pd.DataFrame(final_report)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    run_optimized_backtest()
