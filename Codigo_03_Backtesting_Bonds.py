import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
FILE_RESULTADOS = "Resultados_Regime.csv"
FILE_SHY = "SHY_Historical.csv"
CAPITAL_INICIAL = 100000.0

def clean_yfinance_csv(path):
    """Limpia el CSV de yfinance que suele tener cabeceras complejas"""
    # Intentamos leer saltando posibles líneas de metadatos
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    
    # Colapsamos el multi-index si existe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(c).lower() for c in df.columns]
    
    # Aseguramos que Close sea numérico
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna(subset=['close'])

def run_bond_backtest():
    print("Iniciando Auditoría de Renta Fija (Bolsa + SHY)...")
    
    if not Path(FILE_RESULTADOS).exists():
        print(f"Error: No se encuentra {FILE_RESULTADOS}")
        return

    # 1. Cargar datos
    df_trades = pd.read_csv(FILE_RESULTADOS)
    df_shy = clean_yfinance_csv(FILE_SHY)
    
    # Calcular retornos diarios de SHY
    df_shy['retorno'] = df_shy['close'].pct_change().fillna(0)
    
    # 2. PnL de la Bolsa (Datos del TFM)
    pnl_bolsa = df_trades['pnl_neto'].sum()
    roi_bolsa = (pnl_bolsa / CAPITAL_INICIAL) * 100
    
    # 3. Simulación de Renta Fija (El "Efecto SHY")
    # Asumimos que el 60% del capital está en bonos de corto plazo como 'colchón'
    cash_idle = CAPITAL_INICIAL * 0.60
    
    # Rango temporal de los trades
    df_trades['entry_ts'] = pd.to_datetime(df_trades['entry_ts'])
    start_date = df_trades['entry_ts'].min()
    end_date = df_trades['entry_ts'].max()
    
    # Filtramos SHY y calculamos el crecimiento compuesto
    shy_period = df_shy.loc[start_date:end_date]
    if shy_period.empty:
        # Si hay desfase de fechas, usamos el retorno medio histórico (~2% anual)
        print("Aviso: Usando estimación conservadora por desfase de fechas.")
        años = (end_date - start_date).days / 365
        retorno_shy_total = (1 + 0.02) ** años - 1
    else:
        retorno_shy_total = (1 + shy_period['retorno']).prod() - 1
    
    pnl_extra_bonds = cash_idle * retorno_shy_total
    
    # 4. Resultados finales
    total_pnl = pnl_bolsa + pnl_extra_bonds
    total_roi = (total_pnl / CAPITAL_INICIAL) * 100
    
    print("\n" + "="*40)
    print("      INFORME DE OPTIMIZACIÓN GLOBAL")
    print("="*40)
    print(f"ROI Estrategia Bolsa:   {roi_bolsa:10.2f}%")
    print(f"ROI Extra Bonos (SHY):  { (pnl_extra_bonds/CAPITAL_INICIAL)*100:10.2f}%")
    print("-" * 40)
    print(f"ROI TOTAL SINCRONIZADO: {total_roi:10.2f}%")
    print("="*40)
    print(f"INCREMENTO DE VALOR:    {total_roi - roi_bolsa:10.2f}%")
    
    # Guardar resultados
    with open("RESULTADOS_FINALES_DIVERSIFICADOS.txt", "w", encoding="utf-8") as f:
        f.write(f"SISTEMA DE TRADING IA - RESULTADOS FINALES\n")
        f.write(f"------------------------------------------\n")
        f.write(f"ROI Bolsa (Multiactivo): {roi_bolsa:.2f}%\n")
        f.write(f"ROI Renta Fija (SHY):    {(pnl_extra_bonds/CAPITAL_INICIAL)*100:.2f}%\n")
        f.write(f"ROI TOTAL COMBINADO:     {total_roi:.2f}%\n")

if __name__ == "__main__":
    run_bond_backtest()
