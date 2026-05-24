file_path = r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\HISTORICOS\Historicos IBEX 35 19052026.txt"

factors = set()
count = 0
with open(file_path, 'r', encoding='latin1') as f:
    for line in f:
        if line.startswith("0001"):
            parts = line.split(",")
            if len(parts) > 9:
                adj = float(parts[9].strip())
                factors.add(adj)
                if adj != 1.0 and count < 10:
                    print(f"Non-1.0 adj: {line.strip()}")
                    count += 1

print("Unique adjustment factors found:", sorted(list(factors)))
