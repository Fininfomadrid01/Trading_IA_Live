import pandas as pd

ibex_isins = pd.read_csv(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\isins_ibex35.csv")
print("isins_ibex35.csv columns:", ibex_isins.columns.tolist())
print(ibex_isins.head(2))

mc_isins = pd.read_csv(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\isins_mercadocontinuo.csv")
print("\nisins_mercadocontinuo.csv columns:", mc_isins.columns.tolist())
print(mc_isins.head(2))

# Let's try to map the unique codes from the Historicos IBEX 35 19052026.txt
file_path = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\HISTORICOS\Historicos IBEX 35 19052026.txt"

all_codes = set()
with open(file_path, 'r', encoding='latin1') as f:
    for line in f:
        if line.startswith("0001"):
            parts = line.split(",")
            if len(parts) > 1:
                # Format is: 0001,XXES0111845014            055, ...
                raw_code = parts[1].strip()
                # Split raw_code by space to separate ISIN and code
                subparts = [p for p in raw_code.split() if p]
                if subparts:
                    isin = subparts[0]
                    # Sometimes ISIN starts with XX, let's clean it
                    if isin.startswith("XX") or isin.startswith("YY") or isin.startswith("ZZ"):
                        isin = isin[2:]
                    all_codes.add(isin)

print(f"\nFound {len(all_codes)} unique cleaned ISINs in Historicos IBEX 35 file.")

# Normalize ISIN column name
ibex_isins = ibex_isins.rename(columns={'CVALISO': 'isin'})
mc_isins = mc_isins.rename(columns={'ISIN': 'isin'})

# Strip whitespaces from isin and VALOR
ibex_isins['isin'] = ibex_isins['isin'].astype(str).str.strip()
mc_isins['isin'] = mc_isins['isin'].astype(str).str.strip()
ibex_isins['VALOR'] = ibex_isins['VALOR'].astype(str).str.strip()
mc_isins['VALOR'] = mc_isins['VALOR'].astype(str).str.strip()

print("Mapping clean ISINs with isins_ibex35.csv:")
mapped_ibex = ibex_isins[ibex_isins['isin'].isin(all_codes)]
print(f"Mapped in IBEX 35: {len(mapped_ibex)} of {len(all_codes)}")
print(mapped_ibex)

print("\nMapping clean ISINs with isins_mercadocontinuo.csv:")
mapped_mc = mc_isins[mc_isins['isin'].isin(all_codes)]
print(f"Mapped in Mercado Continuo: {len(mapped_mc)} of {len(all_codes)}")
print(mapped_mc)

# Find unmapped
mapped_isins = set(mapped_ibex['isin']).union(set(mapped_mc['isin']))
unmapped = all_codes - mapped_isins
print("\nUnmapped clean ISINs:", unmapped)
