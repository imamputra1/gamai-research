import numpy as np
from pathlib import Path

def load_bootstrap_data(filename="bootstrap_array.npy"):
    """
    Memuat file .npy dari direktori reports/blockD/tables/
    """
    # Menentukan path absolut relatif terhadap file ini
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    data_path = root_dir / "reports" / "analisis" / "tables" / filename

    if data_path.exists():
        data = np.load(data_path)
        print(f"Berhasil dimuat: {filename} | Shape: {data.shape}")
        return data
    else:
        print(f"Error: File tidak ditemukan di {data_path}")
        return None

# Cara penggunaan:
data = load_bootstrap_data()

