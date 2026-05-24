import re

def main():
    with open('Codigo_10_Lanzador_IBKR.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Fix get_market_config
    old_config = """def get_market_config(ticker):
    if ticker.endswith(".MC"): return MARKET_MAP['IBEX']
    if ticker.endswith(".L"):  return MARKET_MAP['FTSE']
    if ticker.endswith("-USD"): return MARKET_MAP['CRYPTO']
    if ticker.endswith("=F"):   return MARKET_MAP['COMMO']
    return MARKET_MAP['USA']"""

    new_config = """USA_TICKERS = ['AAPL', 'AMD', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NFLX', 'NVDA', 'TSLA', 'QQQ', 'GOLD', 'OIL', 'BTC', 'ETH']

def get_market_config(ticker):
    if ticker.endswith(".L"):  return MARKET_MAP['FTSE']
    if ticker.endswith("-USD"): return MARKET_MAP['CRYPTO']
    if ticker.endswith("=F"):   return MARKET_MAP['COMMO']
    if ticker in USA_TICKERS: return MARKET_MAP['USA']
    return MARKET_MAP['IBEX']"""

    content = content.replace(old_config, new_config)

    # 2. Fix the placeOrder logic so IDs are generated properly.
    old_order_logic = """        # 1. Orden de Compra (Parent)
        parent = MarketOrder('BUY', cantidad)
        parent.transmit = False

        # 2. Orden de Trailing Stop Optimizado (Child)
        trail = Order()
        trail.action = 'SELL'
        trail.orderType = 'TRAIL'
        trail.totalQuantity = cantidad
        trail.trailingPercent = cfg['stop']
        trail.parentId = parent.orderId
        trail.transmit = True

        if DRY_RUN:
            print(f"   >>> [MODO PRUEBA] Simulacin completa para {tk}:")
            print(f"       Accin: COMPRA {cantidad} unidades")
            print(f"       Instrumento: {contract.symbol} ({contract.secType})")
            print(f"       Trailing Stop: {cfg['stop']}%")
            continue

        ib.placeOrder(contract, parent)
        ib.placeOrder(contract, trail)
        print(f" [OK] Orden enviada: {tk} (Trailing Stop: {cfg['stop']}%)")
        time.sleep(0.5)"""

    new_order_logic = """        # 1. Generar IDs nicos
        parent_id = ib.client.getReqId()
        
        # 2. Orden de Compra (Parent)
        parent = MarketOrder('BUY', cantidad)
        parent.orderId = parent_id
        parent.transmit = False

        # 3. Orden de Trailing Stop Optimizado (Child)
        trail = Order()
        trail.orderId = ib.client.getReqId()
        trail.action = 'SELL'
        trail.orderType = 'TRAIL'
        trail.totalQuantity = cantidad
        trail.trailingPercent = cfg['stop']
        trail.parentId = parent_id
        trail.transmit = True

        if DRY_RUN:
            print(f"   >>> [MODO PRUEBA] Simulacin completa para {tk}:")
            print(f"       Accin: COMPRA {cantidad} unidades")
            print(f"       Instrumento: {contract.symbol} ({contract.secType})")
            print(f"       Trailing Stop: {cfg['stop']}%")
            continue

        ib.placeOrder(contract, parent)
        ib.placeOrder(contract, trail)
        print(f" [OK] Orden enviada: {tk} (Trailing Stop: {cfg['stop']}%)")
        time.sleep(0.5)"""

    content = content.replace(old_order_logic, new_order_logic)
    
    with open('Codigo_10_Lanzador_IBKR.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Correcciones aplicadas exitosamente.")

if __name__ == "__main__":
    main()
