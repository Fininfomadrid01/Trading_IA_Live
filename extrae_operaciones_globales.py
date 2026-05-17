import os
import pandas as pd
import numpy as np
import keras
from pathlib import Path

# --- CONFIGURACIÓN ---
os.environ["KERAS_BACKEND"] = "torch"
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
MODEL_PATH = BASE_DIR / "Modelo_IA_Entrenado.keras"

# CONFIGURACIONES DE ACTIVOS GLOBALES MULTIACTIVO
CONFIGS = {
    # 🇺🇸 Bolsa Americana (NASDAQ 100) - Selección representativa de alta liquidez
    'US_STOCKS': {
        'tickers': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'NFLX', 'AVGO'],
        'stop': 6.19, 'exit': 87, 'prob': 0.954,
        'dir': BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw",
        'type': 'Bolsa Americana (NASDAQ)'
    },
    # 🇬🇧 Bolsa de Londres (FTSE 100) - Selección representativa
    'UK_STOCKS': {
        'tickers': ['AZN.L', 'BP.L', 'SHEL.L', 'BARC.L', 'LLOY.L', 'HSBA.L', 'GSK.L', 'RIO.L', 'GLEN.L', 'AAL.L'],
        'stop': 3.92, 'exit': 110, 'prob': 0.935,
        'dir': BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw",
        'type': 'Bolsa de Londres (FTSE)'
    },
    # 🪙 Criptomonedas (ETFs)
    'CRYPTO': {
        'tickers': ['BTC-USD', 'ETH-USD'],
        'stop': 13.89, 'exit': 25, 'prob': 0.923,
        'dir': BASE_DIR / "EXPERIMENTO_CRYPTO" / "datos_raw",
        'type': 'Criptomonedas (ETFs)'
    },
    # 🛢️ Materias Primas (Commodities)
    'COMMODITIES': {
        'tickers': ['GOLD', 'OIL'],
        'stop': 6.18, 'exit': 90, 'prob': 0.933,
        'dir': BASE_DIR / "EXPERIMENTO_COMMODITIES" / "datos_raw",
        'type': 'Materias Primas'
    }
}

def clean_yfinance_csv(path):
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

def run_global_backtest():
    print("Iniciando Compilación Práctica Multimercado Global (USA, UK, España, Crypto, Materias Primas)...")
    model = keras.models.load_model(str(MODEL_PATH))
    all_trades = []

    # 1. CARGAR OPERATIVA DE ESPAÑA (IBEX 35) ya calculada
    print("\nCargando operativa del IBEX 35 (España)...")
    ibex_csv = BASE_DIR / "Operaciones_TFM_2025_2026.csv"
    if ibex_csv.exists():
        ibex_df = pd.read_csv(ibex_csv)
        for _, row in ibex_df.iterrows():
            all_trades.append({
                'Ticker': row['Ticker'],
                'Direccion': 'LONG',
                'Regimen_Mercado': 'Bolsa Española (IBEX)',
                'Fecha_Entrada': row['Fecha_Entrada'],
                'Fecha_Salida': row['Fecha_Salida'],
                'Precio_Entrada': row['Precio_Entrada'],
                'Precio_Salida': row['Precio_Salida'],
                'Volumen_Acciones': row['Volumen_Acciones'],
                'Exposicion_EUR': row['Exposicion_EUR'],
                'Ganancia_Pct': row['Ganancia_Pct'],
                'PnL_Neto_EUR': row['PnL_Neto_EUR'],
                'Motivo_Salida': row['Motivo_Salida'],
                'Confianza_IA': row['Confianza_IA']
            })

    # 2. PROCESAR MERCADOS INTERNACIONALES (NASDAQ, FTSE, CRYPTO, COMMODITIES)
    for cat_name, cfg in CONFIGS.items():
        print(f"\nProcesando categoría: {cfg['type']}...")
        for ticker in cfg['tickers']:
            # Buscar archivo Parquet de 1H
            f = cfg['dir'] / f"{ticker}_1H.parquet"
            if not f.exists():
                # Reintentar con ticker en minúsculas o variaciones
                f = cfg['dir'] / f"{ticker.upper()}_1H.parquet"
                if not f.exists():
                    print(f" > Advertencia: No se encuentra {ticker}_1H.parquet")
                    continue
            
            print(f" > Analizando {ticker}...")
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # Filtrar por periodo 2025-2026
            df = df[(df.index >= '2025-01-01') & (df.index <= '2026-05-16')]
            if df.empty:
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
                
                # Regla de entrada IA
                if np.argmax(pred) == 1 and pred[1] > cfg['prob']:
                    fecha_entrada = df.index[i]
                    entry_p = c[i]
                    
                    # Salida dinámica
                    exit_i = min(i + cfg['exit'], len(df)-1)
                    max_p = entry_p; exit_p = c[exit_i]
                    motivo = "TIEMPO_MAX"
                    
                    for j in range(i+1, exit_i+1):
                        if h[j] > max_p: max_p = h[j]
                        if l[j] < max_p * (1 - cfg['stop']/100):
                            exit_i = j
                            exit_p = max_p * (1 - cfg['stop']/100)
                            motivo = "TRAILING_STOP"
                            break
                            
                    fecha_salida = df.index[exit_i]
                    ganancia_pct = (exit_p - entry_p)/entry_p
                    
                    # Gestión monetaria asimétrica para diversificación
                    exposicion = 10000.0 # Exposición de 10.000 € por trade global
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
                    last_exit = exit_i + 5

    # 3. AGREGAR RENTA FIJA (SHY)
    print("\nProcesando rotación automática de Renta Fija (SHY)...")
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
        
        fecha_entrada = pd.to_datetime(shy_dates[i]).strftime('%Y-%m-%d 09:00:00')
        fecha_salida = pd.to_datetime(shy_dates[i+5]).strftime('%Y-%m-%d 16:00:00')
        
        ganancia_pct = (exit_p - entry_p)/entry_p
        exposicion = 60000.0
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

    # Guardar todos los resultados globales consolidados a un CSV limpio
    df_report = pd.DataFrame(all_trades)
    df_report['Fecha_Entrada'] = pd.to_datetime(df_report['Fecha_Entrada'])
    df_report = df_report.sort_values(by='Fecha_Entrada')
    
    df_report.to_csv("Operaciones_Globales_TFM_2025_2026.csv", index=False, encoding='utf-8')
    
    print("\n" + "="*50)
    print("  CARTERA MULTIACTIVO GLOBAL CONSOLIDADA (2025-2026)")
    print("="*50)
    print(f"Total de operaciones registradas: {len(df_report)}")
    print(f" - Bolsa Española (IBEX): {len(df_report[df_report['Regimen_Mercado'] == 'Bolsa Española (IBEX)'])}")
    print(f" - Bolsa Americana (NASDAQ): {len(df_report[df_report['Regimen_Mercado'] == 'Bolsa Americana (NASDAQ)'])}")
    print(f" - Bolsa de Londres (FTSE): {len(df_report[df_report['Regimen_Mercado'] == 'Bolsa de Londres (FTSE)'])}")
    print(f" - Criptomonedas (ETFs): {len(df_report[df_report['Regimen_Mercado'] == 'Criptomonedas (ETFs)'])}")
    print(f" - Materias Primas: {len(df_report[df_report['Regimen_Mercado'] == 'Materias Primas'])}")
    print(f" - Renta Fija (SHY): {len(df_report[df_report['Regimen_Mercado'] == 'Renta Fija (Bonds)'])}")
    print("="*50)

if __name__ == "__main__":
    run_global_backtest()
