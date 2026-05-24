file_path = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\HISTORICOS\Historicos IBEX 35 19052026.txt"

print("All lines starting with 8000:")
with open(file_path, 'r', encoding='latin1') as f:
    for line in f:
        if line.startswith("8000"):
            print(line.strip())
