from ib_insync import *

ib = IB()

def test_data():
    try:
        # Conectamos (usamos un clientId diferente para no chocar)
        ib.connect('127.0.0.1', 7497, clientId=99)
        print("Conectado a TWS...")

        # Definimos el contrato de Santander en la Bolsa de Madrid (BM)
        contract = Stock('SAN', 'BM', 'EUR')
        ib.qualifyContracts(contract)

        # Pedimos velas diarias de la última semana
        print(f"Pidiendo datos para {contract.symbol}...")
        bars = ib.reqHistoricalData(
            contract, 
            endDateTime='', 
            durationStr='5 D',
            barSizeSetting='1 day', 
            whatToShow='TRADES', 
            useRTH=True
        )

        if bars:
            df = util.df(bars)
            print("\nEXITO! Aqui tienes los ultimos datos de IBKR:")
            print(df[['date', 'open', 'high', 'low', 'close', 'volume']])
        else:
            print("\nAVISO: No se han recibido datos. Posiblemente por falta de suscripcion o mercado cerrado sin datos diferidos permitidos por API.")

        ib.disconnect()

    except Exception as e:
        print(f"Error en la prueba: {e}")

if __name__ == "__main__":
    test_data()
