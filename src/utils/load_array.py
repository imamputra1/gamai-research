import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parents[2]
data_path = root_dir / "reports" / "blockD" / "tables" /"bootstrap_array.npy"

if data_path.exists():
    data = np.load(data_path)
    print("Array berhasil dimuat!", data.shape, "\n", data)
else:
    print(f"Error: File tidak ditemukan di {data_path}")

