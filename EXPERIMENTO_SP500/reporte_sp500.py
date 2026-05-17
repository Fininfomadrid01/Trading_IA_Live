import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
RESULTS_FILE = BASE_DIR / "EXPERIMENTO_SP500" / "resultados" / "trades_sp500.csv"
DATA_DIR = BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw"

def analyze_results():
    if not RESULTS_FILE.exists():
        print("No se han encontrado resultados. Ejecuta primero el motor.")
        return
        
    df = pd.read_csv(RESULTS_FILE)
    df['entry_dt'] = pd.to_datetime(df['entry_dt'])
    df['exit_dt'] = pd.to_datetime(df['exit_dt'])
    
    spy = pd.read_parquet(DATA_DIR / "SPY_1D.parquet")
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    
    start_date = df['entry_dt'].min().tz_localize(None)
    end_date = df['exit_dt'].max().tz_localize(None)
    
    try:
        s_start = float(spy.asof(start_date)['Close'])
        s_end = float(spy.asof(end_date)['Close'])
        roi_spy = float((s_end - s_start) / s_start)
    except:
        roi_spy = 0.0
    
    win_rate = (df['gain'] > 0).mean()
    avg_gain = df['gain'].mean()
    roi_sistema = df['gain'].sum() / 40 # Mayor diversificación en S&P 500
    
    gains = df[df['gain'] > 0]['gain'].sum()
    losses = abs(df[df['gain'] < 0]['gain'].sum())
    pf = gains / losses if losses > 0 else np.inf

    print(f"\n====================================================")
    print(f"      REPORTE FINAL: EXPERIMENTO S&P 500")
    print(f"====================================================")
    print(f"Periodo Analizado:  {start_date.date()} a {end_date.date()}")
    print(f"Total Operaciones:  {len(df)}")
    print(f"Win Rate:           {win_rate:.2%}")
    print(f"Profit Factor:      {pf:.2f}")
    print(f"Ganancia Media:     {avg_gain:.2%}")
    print(f"----------------------------------------------------")
    print(f"ROI S&P 500 (SPY):  {roi_spy:.2%}")
    print(f"ROI SISTEMA IA:     {roi_sistema:.2%}")
    print(f"====================================================")

if __name__ == "__main__":
    analyze_results()
