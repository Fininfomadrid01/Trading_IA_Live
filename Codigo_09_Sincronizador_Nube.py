import requests
from pathlib import Path

# --- CONFIGURACIÓN ---
GITHUB_RAW_URL = "https://raw.githubusercontent.com/Fininfomadrid01/Trading_IA_Live/main/cache_predicciones_LIVE.npz"
LOCAL_DESTINY = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\evaluacion_algoritmos\resultados\cache_predicciones_fixed\cache_predicciones_LIVE.npz")

def descargar_actualizacion():
    print(f"Conectando con GitHub para descargar base de datos LIVE...")
    try:
        response = requests.get(GITHUB_RAW_URL)
        if response.status_code == 200:
            LOCAL_DESTINY.parent.mkdir(parents=True, exist_ok=True)
            with open(LOCAL_DESTINY, "wb") as f:
                f.write(response.content)
            print(f"Sincronizacion completada!")
            print(f"Archivo guardado en: {LOCAL_DESTINY}")
        else:
            print(f"Error al descargar: Codigo {response.status_code}")
    except Exception as e:
        print(f"Error de conexion: {e}")

if __name__ == "__main__":
    descargar_actualizacion()
