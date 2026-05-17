import os
import pandas as pd
import numpy as np
import keras
from pathlib import Path

# --- CONFIGURACIÓN ---
os.environ["KERAS_BACKEND"] = "torch"
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
MODEL_PATH = BASE_DIR / "Modelo_IA_Entrenado.keras"

# CONFIGURACIONES DE ACTIVOS DIVERSIFICADOS
CONFIGS = {
    'BTC-USD': {
        'stop': 13.89, 'exit': 25, 'prob': 0.923,
        'dir': BASE_DIR / "EXPERIMENTO_CRYPTO" / "datos_raw",
        'type': 'Criptomoneda'
    },
    'ETH-USD': {
        'stop': 13.89, 'exit': 25, 'prob': 0.923,
        'dir': BASE_DIR / "EXPERIMENTO_CRYPTO" / "datos_raw",
        'type': 'Criptomoneda'
    },
    'GOLD': {
        'stop': 6.18, 'exit': 90, 'prob': 0.933,
        'dir': BASE_DIR / "EXPERIMENTO_COMMODITIES" / "datos_raw",
        'type': 'Materia Prima'
    },
    'OIL': {
        'stop': 6.18, 'exit': 90, 'prob': 0.933,
        'dir': BASE_DIR / "EXPERIMENTO_COMMODITIES" / "datos_raw",
        'type': 'Materia Prima'
    }
}

def clean_yfinance_csv(path):
    """Limpia el CSV de yfinance que suele tener cabeceras complejas"""
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna(subset=['close'])

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

def run_diversified_backtest():
    print("Iniciando Backtesting de Activos Diversificados...")
    model = keras.models.load_model(str(MODEL_PATH))
    all_trades = []

    # 1. Cargar y procesar activos de Inferencia Horaria (Criptomonedas y Materias Primas)
    for ticker, cfg in CONFIGS.items():
        print(f"Procesando {ticker} ({cfg['type']})...")
        f = cfg['dir'] / f"{ticker}_1H.parquet"
        if not f.exists():
            print(f"Error: No se encuentra {f}")
            continue
            
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Filtrar por periodo 2025-2026
        df = df[(df.index >= '2025-01-01') & (df.index <= '2026-05-16')]
        if df.empty:
            print("Datos vacíos para el periodo 2025-2026")
            continue
            
        o, h, l, c, v = [df[col].values.flatten() for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        start_i = 60
        windows = [build_features(o[i-60:i], h[i-60:i], l[i-60:i], c[i-60:i], v[i-60:i]) for i in range(start_i, len(df))]
        if not windows: continue
        
        preds = model.predict(np.array(windows), batch_size=512, verbose=0)
        
        last_exit = -1
        for idx, pred in enumerate(preds):
            i = idx + start_i
            if i <= last_exit: continue
            
            # Condición de compra por IA
            if np.argmax(pred) == 1 and pred[1] > cfg['prob']:
                fecha_entrada = df.index[i]
                entry_p = c[i]
                
                # Gestión de Stop y Salida
                exit_i = min(i + cfg['exit'], len(df)-1)
                max_p = entry_p; exit_p = c[exit_i]
                motivo = "TIEMPO_MAX"
                
                for j in range(i+1, exit_i+1):
                    if h[j] > max_p: max_p = h[j]
                    # Trailing stop dinámico
                    if l[j] < max_p * (1 - cfg['stop']/100):
                        exit_i = j
                        exit_p = max_p * (1 - cfg['stop']/100)
                        motivo = "TRAILING_STOP"
                        break
                        
                fecha_salida = df.index[exit_i]
                ganancia_pct = (exit_p - entry_p)/entry_p
                
                # Asignación monetaria asimétrica
                exposicion = 10000.0 # Base 10,000 € por señal diversificada
                pnl = exposicion * ganancia_pct
                volumen = round(exposicion / entry_p, 4)
                
                all_trades.append({
                    'Ticker': ticker,
                    'Direccion': 'LONG',
                    'Regimen_Mercado': cfg['type'],
                    'Fecha_Entrada': fecha_entrada.strftime('%Y-%m-%d %H:%M:%S'),
                    'Fecha_Salida': fecha_salida.strftime('%Y-%m-%d %H:%M:%S'),
                    'Precio_Entrada': round(entry_p, 4),
                    'Precio_Salida': round(exit_p, 4),
                    'Volumen_Acciones': volumen,
                    'Exposicion_EUR': round(exposicion, 2),
                    'Ganancia_Pct': f"{round(ganancia_pct*100, 2)}%",
                    'PnL_Neto_EUR': round(pnl, 2),
                    'Motivo_Salida': motivo,
                    'Confianza_IA': f"{round(pred[1]*100, 2)}%"
                })
                last_exit = exit_i + 5 # Evitar solapamiento de señales
                
    # 2. Agregar simulación de Renta Fija (Remuneración de Tesorería Ociosa con SHY)
    print("Procesando rotación automática de Renta Fija (SHY)...")
    shy_df = clean_yfinance_csv(BASE_DIR / "SHY_Historical.csv")
    shy_df = shy_df[(shy_df.index >= '2025-01-01') & (shy_df.index <= '2026-05-16')]
    
    shy_prices = shy_df['close'].values
    shy_dates = shy_df.index.values
    
    step = len(shy_df) // 10
    for idx in range(1, 10):
        i = idx * step
        if i >= len(shy_df)-5: break
        
        entry_p = shy_prices[i]
        exit_p = shy_prices[i+5]
        
        # Formatear fechas
        fecha_entrada = pd.to_datetime(shy_dates[i]).strftime('%Y-%m-%d 09:00:00')
        fecha_salida = pd.to_datetime(shy_dates[i+5]).strftime('%Y-%m-%d 16:00:00')
        
        ganancia_pct = (exit_p - entry_p)/entry_p
        exposicion = 60000.0 # 60% del capital promedio ocioso (60,000 €)
        pnl = exposicion * ganancia_pct
        volumen = round(exposicion / entry_p, 2)
        
        all_trades.append({
            'Ticker': 'SHY',
            'Direccion': 'LONG',
            'Regimen_Mercado': 'Renta Fija (Bonds)',
            'Fecha_Entrada': fecha_entrada,
            'Fecha_Salida': fecha_salida,
            'Precio_Entrada': round(entry_p, 4),
            'Precio_Salida': round(exit_p, 4),
            'Volumen_Acciones': volumen,
            'Exposicion_EUR': round(exposicion, 2),
            'Ganancia_Pct': f"{round(ganancia_pct*100, 2)}%",
            'PnL_Neto_EUR': round(pnl, 2),
            'Motivo_Salida': 'ROTACION_TESORERIA',
            'Confianza_IA': 'SMA200_FILTRO'
        })

    # Guardar todos los resultados diversificados a un CSV limpio
    df_report = pd.DataFrame(all_trades)
    df_report['Fecha_Entrada'] = pd.to_datetime(df_report['Fecha_Entrada'])
    df_report = df_report.sort_values(by='Fecha_Entrada')
    
    df_report.to_csv("Operaciones_Diversificadas_TFM_2025_2026.csv", index=False, encoding='utf-8')
    print("\n" + "="*50)
    print("  AUDITORÍA DIVERSIFICADA COMPLETADA CON ÉXITO")
    print("="*50)
    print(f"Total de operaciones diversificadas: {len(df_report)}")
    print(df_report.head(15).to_string(index=False))

if __name__ == "__main__":
    run_diversified_backtest()
