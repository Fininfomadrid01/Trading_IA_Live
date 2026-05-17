import yfinance as yf
import pandas as pd
import os
from pathlib import Path

# Configuración
OUT_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\EXPERIMENTO_NASDAQ\datos_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_nasdaq100_tickers():
    # Usamos una lista estática fiable de los componentes actuales del NASDAQ 100
    # En un entorno real, esto se bajaría de Wikipedia o Nasdaq.com
    tickers = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "TSLA", "META", "AVGO", "PEP",
        "COST", "ADBE", "CSCO", "NFLX", "AMD", "CMCSA", "TMUS", "INTC", "TXN", "AMGN",
        "HON", "QCOM", "INTU", "SBUX", "AMAT", "ISRG", "MDLZ", "GILD", "BKNG", "ADI",
        "ADP", "VRTX", "REGN", "PYPL", "PANW", "SNPS", "MU", "CDNS", "CSX", "ASML",
        "MAR", "MELI", "LRCX", "ORLY", "CTAS", "KLAC", "MNST", "KDP", "FTNT", "CHTR",
        "ADSK", "AEP", "PDD", "CPRT", "PAYX", "PCAR", "DXCM", "IDXX", "ODFL", "LULU",
        "GEHC", "EXC", "ROST", "MRVL", "AZN", "BKR", "KLA", "MCHP", "WDAY", "ON",
        "CTSH", "TEAM", "ADX", "ABNB", "BIIB", "DDOG", "CEG", "GFS", "MDB", "VRSK",
        "CSGP", "FAST", "ZS", "EBAY", "ANSS", "ENPH", "ALGN", "WBD", "ILMN", "JD"
    ]
    return tickers

def download_data():
    tickers = get_nasdaq100_tickers()
    print(f"Iniciando descarga de {len(tickers)} activos del NASDAQ 100...")
    
    # También bajamos el QQQ como referencia de régimen (Benchmark)
    tickers.append("QQQ")
    
    for symbol in tickers:
        try:
            print(f"Descargando {symbol}...")
            # Bajamos 1H (máximo permitido por YF es 730d)
            data = yf.download(symbol, period="730d", interval="1h", progress=False)
            if not data.empty:
                data.to_parquet(OUT_DIR / f"{symbol}_1H.parquet")
            
            # Bajamos 1D (desde 2018)
            data_daily = yf.download(symbol, start="2018-01-01", interval="1d", progress=False)
            if not data_daily.empty:
                data_daily.to_parquet(OUT_DIR / f"{symbol}_1D.parquet")
                
        except Exception as e:
            print(f"Error con {symbol}: {e}")

if __name__ == "__main__":
    download_data()
