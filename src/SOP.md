# PACKAGE_ARCHITECTURE: src/
> **Tujuan:** SOP Interaksi Antar-Modul & Peta Aliran Data (*Data Flow*)
> **Aturan AI Agent:** Baca file ini sebelum me-refactor variabel atau memindahkan fungsi antar folder.

## 1. PEMETAAN DOMAIN SUB-PACKAGE (SEPARATION OF CONCERNS)
<domain_mapping>

| Nama Sub-Package | Peran & Tanggung Jawab Utama | Batasan Mutlak (*Strict Constraints*) |
| :--- | :--- | :--- |
| **`core_reproducibility`** | Membangun infrastruktur awal (folder `data/`, `reports/`) dan inisialisasi `pipeline_config.yaml`. | Hanya dieksekusi di Fase 0. Jangan letakkan manipulasi data di sini. |
| **`core_synthesis`** | Mesin utama Injeksi DNA (Cholesky), ekspansi Likert (Monte Carlo), dan sintesis teks (OpenRouter). | **HANYA** menyimpan *output* ke `data/03_synthetic/`. Dilarang memodifikasi file *raw*. |
| **`core_analysis`** | Eksekusi Uji Kualitas Data (Validitas, Reliabilitas) dan Asumsi Klasik (Normalitas, Multikol, Hetero). | **WAJIB** membaca data dari `data/04_report_master/df_report_master.csv`. |
| **`core_regression`** | Menghitung *Path Coefficient* OLS untuk Substruktur 1 (X $\rightarrow$ M) dan Substruktur 2 (X, M $\rightarrow$ Y). | Hanya berurusan dengan jalur langsung (*Direct Effect*). |
| **`core_mediation`** | Menjalankan *Resampling Bootstrap* (5000 iterasi) untuk membuktikan efek *Experience Value* (H4). | Parameter iterasi wajib dibaca dari konfigurasi YAML, jangan di-*hardcode*. |
| **`core_hypothesis`** | Menyatukan output dari regresi dan mediasi untuk mengevaluasi status hipotesis (Alpha < 0.05). | Dilarang mengubah angka P-Value. Hanya mencetak status "Diterima/Ditolak" ke Excel. |
| **`ai_integration`** | Menyimpan klien murni API (Moonshot/OpenRouter). | Dilarang memasukkan logika prompt medis/tesis di sini (Prompt milik `synthesis_text`). |
| **`utils`** | Fungsi *Global Shared* (Logger, Load YAML, Export Excel, Schema). | Tidak boleh mengimpor fungsi dari `core_*` (Cegah *Circular Import*). |

</domain_mapping>

## 2. ATURAN INTERAKSI ANTAR MODUL (PIPELINE RULES)
<interaction_rules>

| Komponen Arsitektur | SOP Eksekusi & Implementasi Kode |
| :--- | :--- |
| **The Orchestrator Pattern** | Setiap folder `core_*` **WAJIB** memiliki file `orchestrator.py` yang mengimpor semua modul di dalam foldernya dan merangkainya menjadi satu fungsi `run_X_pipeline()`. |
| **No-Init Policy** | Semua file `__init__.py` di dalam setiap folder di dalam `src/` **WAJIB KOSONG 100%**. Penggabungan *namespace* dilakukan di `orchestrator.py`. |
| **Aliran Data Satu Arah** | Modul `core_regression`, `core_mediation`, dan `core_hypothesis` **TIDAK BOLEH** saling memanggil fungsi. Mereka harus dipanggil secara berurutan oleh `main.py` di *root directory*. |
| **Pusat Log (Logger)** | Semua aktivitas harus dicatat melalui `from src.utils.logger import setup_logger`. Identitas *logger* wajib sama dengan nama file Python tersebut. |

</interaction_rules>

## 3. ANTI-PATTERNS (LARANGAN KERAS UNTUK AI AGENT)
<anti_patterns>

| Tindakan Dilarang | Konsekuensi / Risiko | Solusi / Cara Benar |
| :--- | :--- | :--- |
| **Circular Import** | Program mati saat dijalankan (*ImportError*). | Pindahkan fungsi yang digunakan bersama ke folder `utils/`. |
| **Hardcoding Path** | Gagal berjalan di OS berbeda (Windows vs Mac). | Gunakan `from pathlib import Path` dan susun *path* dengan operator slash (`/`). |
| **Manipulasi Data di Evaluator** | Mengurangi integritas penelitian tesis. | Jika status "Ditolak", manipulasi **WAJIB** dilakukan di hulu (`core_synthesis/generator_setup.py`), BUKAN di modul uji hipotesis. |

</anti_patterns>
