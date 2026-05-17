import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")

def check_dates():
    files = [
        BASE_DIR / "EXPERIMENTO_IBEX" / "datos_raw" / "SAN.MC_1H.parquet",
        BASE_DIR / "EXPERIMENTO_CRYPTO" / "datos_raw" / "BTC-USD_1H.parquet"
    ]
    for f in files:
        if f.exists():
            df = pd.read_parquet(f)
            print(f"Archivo: {f.name} | Último dato: {df.index[-1]}")

if __name__ == "__main__":
    check_dates()
