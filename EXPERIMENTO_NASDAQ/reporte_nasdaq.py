import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
RESULTS_FILE = BASE_DIR / "EXPERIMENTO_NASDAQ" / "resultados" / "trades_nasdaq.csv"
DATA_DIR = BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw"

def analyze_results():
    df = pd.read_csv(RESULTS_FILE)
    df['entry_dt'] = pd.to_datetime(df['entry_dt'])
    df['exit_dt'] = pd.to_datetime(df['exit_dt'])
    
    # 1. Cargar QQQ para comparar
    qqq = pd.read_parquet(DATA_DIR / "QQQ_1D.parquet")
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    
    # Periodo del experimento (primer y último trade)
    start_date = df['entry_dt'].min().tz_localize(None)
    end_date = df['exit_dt'].max().tz_localize(None)
    
    # ROI del QQQ (Buy & Hold)
    try:
        # Buscamos el precio más cercano al inicio y al fin
        q_start = float(qqq.asof(start_date)['Close'])
        q_end = float(qqq.asof(end_date)['Close'])
        roi_qqq = float((q_end - q_start) / q_start)
    except Exception as e:
        print(f"Error calculando ROI QQQ: {e}")
        roi_qqq = 0.0
    
    # 2. Métricas del Sistema IA
    total_trades = len(df)
    win_rate = (df['gain'] > 0).mean()
    avg_gain = df['gain'].mean()
    
    # Simulación de Cartera (Simplificada: Equidad acumulada)
    # Asumimos que dividimos el capital en cada trade o usamos un tamaño fijo
    # Para ser comparables con el TFM, usaremos el ROI acumulado ponderado
    roi_sistema = df['gain'].sum() / 10 # Aproximación de diversificación (operando 10% por trade)
    
    # Profit Factor
    gains = df[df['gain'] > 0]['gain'].sum()
    losses = abs(df[df['gain'] < 0]['gain'].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    print(f"\n====================================================")
    print(f"      REPORTE FINAL: EXPERIMENTO NASDAQ 100")
    print(f"====================================================")
    print(f"Periodo Analizado:  {start_date.date()} a {end_date.date()}")
    print(f"Total Operaciones:  {total_trades}")
    print(f"Win Rate:           {win_rate:.2%}")
    print(f"Profit Factor:      {profit_factor:.2f}")
    print(f"Ganancia Media:     {avg_gain:.2%}")
    print(f"----------------------------------------------------")
    print(f"ROI NASDAQ (QQQ):   {roi_qqq:.2%}")
    print(f"ROI SISTEMA IA:     {roi_sistema:.2%}")
    print(f"====================================================")
    
    if roi_sistema > roi_qqq:
        print("RESULTADO: EL MODELO IA BATE AL NASDAQ 100 (Out-of-Sample)")
    else:
        print("RESULTADO: EL NASDAQ 100 HA SIDO MÁS RENTABLE (Buy & Hold)")

if __name__ == "__main__":
    analyze_results()
