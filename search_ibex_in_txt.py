file_path = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\HISTORICOS\Historicos IBEX 35 19052026.txt"

print("Searching for lines containing IBEX or INDICE or INDEX...")
count = 0
with open(file_path, 'r', encoding='latin1') as f:
    for i, line in enumerate(f):
        if any(term in line.upper() for term in ["IBEX", "INDICE", "INDEX"]):
            print(f"Line {i+1}: {line.strip()}")
            count += 1
            if count >= 30:
                break
print(f"Found {count} matching lines.")
