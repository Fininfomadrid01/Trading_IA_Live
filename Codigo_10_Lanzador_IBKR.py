from ib_insync import *
import numpy as np
import pandas as pd
from pathlib import Path
import time

# --- CONFIGURACIÓN ESTRATÉGICA GLOBAL ---
CAPITAL_TOTAL = 100000.0  # Actualizado a 100.000€
TOP_N = 12                # Diversificamos en 12 posiciones para reducir riesgo unitario
CACHE_PATH = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\evaluacion_algoritmos\resultados\cache_predicciones_fixed\cache_predicciones_LIVE.npz")

# --- MAPEO DE CONFIGURACIÓN POR MERCADO ---
MARKET_MAP = {
    'IBEX':   {'suffix': '.MC', 'stop': 5.65, 'prob': 0.923, 'exch': 'BM',    'curr': 'EUR'},
    'FTSE':   {'suffix': '.L',  'stop': 3.92, 'prob': 0.935, 'exch': 'LSE',   'curr': 'GBP'},
    'USA':    {'suffix': '',    'stop': 6.19, 'prob': 0.954, 'exch': 'SMART', 'curr': 'USD'},
    'CRYPTO': {'suffix': '-USD', 'stop': 13.89, 'prob': 0.923, 'exch': 'PAXOS', 'curr': 'USD'},
    'COMMO':  {'suffix': '=F',   'stop': 6.18,  'prob': 0.933, 'exch': 'NYMEX', 'curr': 'USD'}
}

# --- MAPA DE TRADUCCIÓN A INSTRUMENTOS (ETFs/ACCIONES) ---
TRADABLE_MAP = {
    'GOLD': {'symbol': 'GLD',  'secType': 'STK', 'exch': 'SMART', 'curr': 'USD'},
    'OIL':  {'symbol': 'USO',  'secType': 'STK', 'exch': 'SMART', 'curr': 'USD'},
    'BTC':  {'symbol': 'IBIT', 'secType': 'STK', 'exch': 'SMART', 'curr': 'USD'},
    'NDX':  {'symbol': 'QQQ',  'secType': 'STK', 'exch': 'SMART', 'curr': 'USD'}
}

# --- CONEXIÓN IBKR ---
IB_PORT = 7497 
ib = IB()

def connect_ib():
    try:
        ib.connect('127.0.0.1', IB_PORT, clientId=0)
        print("Conectado a Interactive Brokers (Modo Multimercado)")
    except Exception as e:
        print(f"Error al conectar: {e}")
        return False
    return True

def get_market_config(ticker):
    if ticker.endswith(".MC"): return MARKET_MAP['IBEX']
    if ticker.endswith(".L"):  return MARKET_MAP['FTSE']
    if ticker.endswith("-USD"): return MARKET_MAP['CRYPTO']
    if ticker.endswith("=F"):   return MARKET_MAP['COMMO']
    return MARKET_MAP['USA']

def main():
    if not connect_ib(): return
    if not CACHE_PATH.exists(): 
        print("Error: No se encuentra la cache. Sincroniza primero con el Codigo 09."); return

    print("Leyendo base de datos de predicciones globales...")
    data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in data.keys() if k.startswith("preds_")])
    
    signals = []
    for tk in tickers:
        try:
            preds = data[f"preds_{tk}"]
            last_c = data[f"c_{tk}"][-1]
            last_v = data[f"v_{tk}"][-1]
            
            # Obtener configuración específica para este activo
            cfg = get_market_config(tk)
            
            clase = np.argmax(preds[-1])
            prob = preds[-1][clase]
            
            # Aplicar filtro de confianza optimizado por mercado
            if clase == 1 and prob >= cfg['prob']:
                vol_ef = last_c * last_v
                alloc = 0.15 if vol_ef > 40e6 else (0.08 if vol_ef > 10e6 else 0.04)
                signals.append({
                    'tk': tk, 
                    'prob': prob, 
                    'inv': CAPITAL_TOTAL * alloc, 
                    'price': last_c,
                    'cfg': cfg
                })
        except: pass

    # Ordenar por probabilidad y seleccionar las mejores señales del mundo
    top_signals = sorted(signals, key=lambda x: x['prob'], reverse=True)[:TOP_N]

    if not top_signals:
        print("Hoy no hay señales que cumplan los criterios optimizados.")
        return

    print(f"ENVIANDO {len(top_signals)} ORDENES OPTIMIZADAS A TWS...")

    for s in top_signals:
        tk = s['tk']
        cfg = get_market_config(tk)
        
        # LÓGICA DE TRADUCCIÓN (ETFs vs Acciones)
        if tk in TRADABLE_MAP:
            m = TRADABLE_MAP[tk]
            contract = Contract()
            contract.symbol = m['symbol']
            contract.secType = m['secType']
            contract.exchange = m['exch']
            contract.currency = m['curr']
            print(f"🔄 Traduciendo señal {tk} -> Instrumento {m['symbol']} ({m['secType']})")
        else:
            contract = Stock(tk, cfg['exch'], cfg['curr'])
            
        ib.qualifyContracts(contract)
        
        cantidad = int(s['inv'] / s['price'])
        if cantidad <= 0: continue

        print(f" > Preparando {tk} ({cfg['exch']}): {cantidad} acciones a {cfg['stop']}% stop...")

        # 1. Orden de Compra (Parent)
        parent = MarketOrder('BUY', cantidad)
        parent.transmit = False

        # 2. Orden de Trailing Stop Optimizado (Child)
        trail = Order()
        trail.action = 'SELL'
        trail.orderType = 'TRAIL'
        trail.totalQuantity = cantidad
        trail.trailingPercent = cfg['stop'] # <--- PARÁMETRO EVOLUCIONADO
        trail.parentId = parent.orderId
        trail.transmit = True

        ib.placeOrder(contract, parent)
        ib.placeOrder(contract, trail)
        
        print(f"   [OK] {ticker_ib} enviado (Stop: {cfg['stop']}%, Prob: {s['prob']:.3f})")
        time.sleep(0.5)

    print("\nProceso finalizado. Las ordenes están en TWS listas para confirmación.")
    ib.disconnect()

if __name__ == "__main__":
    main()
