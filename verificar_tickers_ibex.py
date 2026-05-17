import requests
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0'}
url = 'https://en.wikipedia.org/wiki/IBEX_35'
response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)
df = tables[2]
tickers = df['Ticker'].tolist()
print(f"Tickers encontrados ({len(tickers)}): {tickers}")
