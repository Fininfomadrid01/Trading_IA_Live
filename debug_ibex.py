import pandas as pd
import requests

headers = {'User-Agent': 'Mozilla/5.0'}
url = 'https://en.wikipedia.org/wiki/IBEX_35'
response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)

for i, table in enumerate(tables):
    print(f"Tabla {i}: {table.columns.tolist()}")
