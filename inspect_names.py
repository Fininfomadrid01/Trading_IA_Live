file_path = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\HISTORICOS\Historicos IBEX 35 19052026.txt"

print("Reading first 100 lines...")
with open(file_path, 'r', encoding='latin1') as f:
    for i in range(100):
        line = f.readline()
        if not line:
            break
        print(f"{i+1}: {line.strip()}")
