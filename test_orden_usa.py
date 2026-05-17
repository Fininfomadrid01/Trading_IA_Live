from ib_insync import *
import time

ib = IB()

def test_usa_id0():
    try:
        # USAMOS EL CLIENT ID 0 (EL MAESTRO)
        ib.connect('127.0.0.1', 7497, clientId=0)
        print("Conectado con ID 0 (MASTER)...")

        contract = Stock('AAPL', 'SMART', 'USD')
        ib.qualifyContracts(contract)

        print(f"Lanzando orden de mercado para 1 accion de {contract.symbol} con ID 0...")

        # 1. Orden de Compra
        parent = MarketOrder('BUY', 1)
        parent.transmit = False

        # 2. Trailing Stop
        trail = Order()
        trail.action = 'SELL'
        trail.orderType = 'TRAIL'
        trail.totalQuantity = 1
        trail.trailingPercent = 5.5
        trail.parentId = parent.orderId
        trail.transmit = True

        ib.placeOrder(contract, parent)
        ib.placeOrder(contract, trail)
        
        print("Orden enviada con ID Maestro! Comprueba si en TWS aparece con el boton o ya activada.")
        time.sleep(2)
        ib.disconnect()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_usa_id0()
