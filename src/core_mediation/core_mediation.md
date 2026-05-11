# DOMAIN_RULES: src/core_mediation/
> **Status:** Mesin Kalkulasi Jalur Tidak Langsung (*Indirect Effect*)
> **Tujuan:** Mengeksekusi *Resampling* untuk membuktikan efek mediasi (H4).

## 1. FILE ROLES
<file_roles>

| Nama File | Tanggung Jawab Mutlak |
| :--- | :--- |
| `bootstrap_core.py` | Melakukan *looping resampling* data, menghitung iterasi perkalian koefisien $(a \times b)$, dan mencari *Confidence Interval* (CI) 95%. |
| `orchestrator.py` | Menjalankan *bootstrap* dan mengekspor hasilnya. |
| `__init__.py` | **WAJIB KOSONG.** |

</file_roles>

## 2. AGENT CODING RULES (CORE MEDIATION)
<agent_rules>
- **Metodologi Non-Parametrik:** DILARANG menggunakan Uji Sobel tradisional karena asumsi normalitas sering terlanggar. WAJIB menggunakan metode *Bootstrapping* sebagai *gold standard* tesis.
- **Config-Driven Loops:** Jumlah iterasi *bootstrap* WAJIB menggunakan angka `jumlah_bootstrap` dari `pipeline_config.yaml` (default: 5000).
- **Penentuan Signifikansi:** Hitung *Lower Bound* (Persentil 2.5) dan *Upper Bound* (Persentil 97.5). Jika rentang ini TIDAK melewati angka 0, mediasi signifikan.
- **Random Seed:** Injeksi `random_state` pada pandas `.sample()` di setiap iterasi menggunakan *seed* dari YAML agar hasil 5000 iterasi ini dapat direproduksi 100% (*deterministic*).
</agent_rules>
