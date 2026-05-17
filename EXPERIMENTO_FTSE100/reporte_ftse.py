import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
RESULTS_FILE = BASE_DIR / "EXPERIMENTO_FTSE100" / "resultados" / "trades_ftse.csv"
DATA_DIR = BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw"

def analyze_results():
    if not RESULTS_FILE.exists():
        print("No se han encontrado resultados. Ejecuta primero el motor.")
        return
        
    df = pd.read_csv(RESULTS_FILE)
    df['entry_dt'] = pd.to_datetime(df['entry_dt'])
    df['exit_dt'] = pd.to_datetime(df['exit_dt'])
    
    bench = pd.read_parquet(DATA_DIR / "^FTSE_1D.parquet")
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    
    start_date = df['entry_dt'].min().tz_localize(None)
    end_date = df['exit_dt'].max().tz_localize(None)
    
    try:
        b_start = float(bench.asof(start_date)['Close'])
        b_end = float(bench.asof(end_date)['Close'])
        roi_bench = float((b_end - b_start) / b_start)
    except:
        roi_bench = 0.0
    
    win_rate = (df['gain'] > 0).mean()
    avg_gain = df['gain'].mean()
    roi_sistema = df['gain'].sum() / 10 # Diversificación similar a IBEX/NASDAQ
    
    gains = df[df['gain'] > 0]['gain'].sum()
    losses = abs(df[df['gain'] < 0]['gain'].sum())
    pf = gains / losses if losses > 0 else np.inf

    print(f"\n====================================================")
    print(f"      REPORTE FINAL: EXPERIMENTO FTSE 100")
    print(f"====================================================")
    print(f"Periodo Analizado:  {start_date.date()} a {end_date.date()}")
    print(f"Total Operaciones:  {len(df)}")
    print(f"Win Rate:           {win_rate:.2%}")
    print(f"Profit Factor:      {pf:.2f}")
    print(f"Ganancia Media:     {avg_gain:.2%}")
    print(f"----------------------------------------------------")
    print(f"ROI FTSE 100 (^FTSE): {roi_bench:.2%}")
    print(f"ROI SISTEMA IA:       {roi_sistema:.2%}")
    print(f"====================================================")

if __name__ == "__main__":
    analyze_results()
