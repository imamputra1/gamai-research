```python
content = """# Thesis Synth Pipeline 🚀

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Framework](https://img.shields.io/badge/framework-uv-purple.svg)](https://github.com/astral-sh/uv)

**Thesis Synth Pipeline** adalah sebuah framework data engineering dan komputasi saintifik yang dirancang khusus untuk mensintesis data skala Likert serta melakukan analisis statistik otomatis untuk tesis S2. Proyek ini menggunakan pendekatan *Hybrid (Modern OOP + Functional Core)* dengan *Screaming Architecture* untuk memastikan kode yang dihasilkan *scalable*, *reproducible*, dan mudah dipahami.

---

## 🧠 Latar Belakang & Tujuan
Proyek ini dibangun untuk menangani masalah keterbatasan data responden (pilot study) dengan melakukan ekspansi data menggunakan metode **Monte Carlo** dan **Cholesky Decomposition** (Injeksi DNA Korelasi). Hasil sintesis kemudian diuji melalui pipeline analisis statistik yang ketat, mencakup:
1. **Sintesis Data:** Ekspansi data dari n=20 menjadi N=Target (misal N=100) dengan menjaga struktur korelasi antar variabel.
2. **Analisis Statistik:** Uji Kualitas Data (Validitas/Reliabilitas) dan Uji Asumsi Klasik.
3. **Pengujian Hipotesis:** Regresi Linear Berganda (OLS) dengan *Robust Standard Errors*.
4. **Analisis Mediasi:** Pembuktian efek mediasi (H4) menggunakan **Bootstrap Resampling** (5000 iterasi).

---

## 🏗️ Arsitektur Proyek (Screaming Architecture)

```text
thesis_synth_pipeline/
├── data/                    # Manajemen Data (Raw, Processed, Synthetic)
├── src/                     # Core Logic (Modular Sub-packages)
│   ├── ai_integration/      # Client API LLM (OpenRouter/Kimi)
│   ├── core_synthesis/      # Mesin Sintesis (Monte Carlo)
│   ├── core_analysis/       # Uji Kualitas & Asumsi Klasik
│   ├── core_regression/     # Estimasi Path Coefficient (OLS)
│   ├── core_mediation/      # Bootstrap Mediasi (H4)
│   ├── core_reproducibility/# Inisialisasi Workspace & Config
│   └── utils/               # Logger, Config Loader, Excel Exporter
├── config/                  # Sentralisasi Parameter (YAML)
├── logs/                    # Audit Trail Eksekusi
├── reports/                 # Output Final (Tabel & Grafik)
└── main.py                  # Entry Point Utama (Orchestrator)

```

---

## 🛠️ Instalasi

Proyek ini menggunakan [uv](https://github.com/astral-sh/uv) untuk manajemen paket yang sangat cepat.

1. **Clone Repositori:**
```bash
git clone [https://github.com/username/thesis_synth_pipeline.git](https://github.com/username/thesis_synth_pipeline.git)
cd thesis_synth_pipeline

```


2. **Install Dependencies:**
```bash
uv sync

```


3. **Setup Environment:**
Buat file `.env` di root directory dan tambahkan API Key jika menggunakan fitur sintesis teks:
```env
OPENROUTER_API_KEY=your_api_key_here

```



---

## 🚀 Panduan Eksekusi

Pipeline dijalankan secara modular menggunakan stasiun komando `main.py`.

### 1. Inisialisasi Workspace

Menyiapkan folder, log, dan memvalidasi konfigurasi.

```bash
uv run python main.py --fase 12.0

```

*Nama Internal:* `initialize_workspace`

### 2. Estimasi Sub-Struktur 1 (X -> M)

Menghitung pengaruh variabel independen terhadap mediator.

```bash
uv run python main.py --fase 12.1

```

*Nama Internal:* `estimate_antecedent_effects`

### 3. Estimasi Sub-Struktur 2 (X, M -> Y)

Menghitung pengaruh independen dan mediator terhadap dependen.

```bash
uv run python main.py --fase 12.2

```

*Nama Internal:* `estimate_mediator_outcomes`

### 4. Evaluasi Hipotesis

Membandingkan P-Value dengan Alpha (0.05).

```bash
uv run python main.py --fase 13.1

```

*Nama Internal:* `validate_hypotheses`

### 5. Bootstrap Mediasi (H4)

Menjalankan 5000 iterasi bootstrap untuk menguji efek mediasi.

```bash
uv run python main.py --fase 13.2

```

*Nama Internal:* `compute_mediation_bootstrap`

---

## ⚙️ Konfigurasi (pipeline_config.yaml)

Semua parameter saintifik dikontrol melalui satu file YAML:

* `n_target`: Jumlah responden yang diinginkan.
* `m_datasets`: Jumlah dataset sintesis yang dihasilkan.
* `signifikansi_alpha`: Ambang batas p-value (default 0.05).
* `arsitektur_model`: Definisi variabel Independen, Mediator, dan Dependen.

---

## 🧪 Prinsip Pengembangan

* **ADHD & Dyslexia Friendly:** Kode dirancang dengan penamaan yang deskriptif, struktur yang modular, dan dokumentasi yang visual untuk mengurangi beban kognitif.
* **TDD (Test-Driven Development):** Memastikan setiap fungsi transformasi data memiliki unit test untuk validasi output.
* **Deterministic:** Penggunaan `random_seed` yang konsisten memastikan hasil simulasi dapat direproduksi 100% oleh peneliti lain.

---

## 📄 Lisensi

[MIT License](https://www.google.com/search?q=LICENSE)
"""
