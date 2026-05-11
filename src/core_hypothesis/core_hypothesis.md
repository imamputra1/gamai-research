# DOMAIN_RULES: src/core_hypothesis/
> **Status:** Hakim Keputusan Final (Evaluator)
> **Tujuan:** Menggabungkan seluruh P-Value & CI, membandingkannya dengan Alpha, lalu mencetak vonis Diterima/Ditolak.

## 1. FILE ROLES
<file_roles>

| Nama File | Tanggung Jawab Mutlak |
| :--- | :--- |
| `evaluator.py` | Membaca hasil `core_regression` dan `core_mediation`, menerapkan logika kondisional statis (`< alpha`), dan menyusun tabel ringkasan hipotesis. |
| `orchestrator.py` | Mengeksekusi evaluator dan menyimpan output ke format Excel. |
| `__init__.py` | **WAJIB KOSONG.** |

</file_roles>

## 2. AGENT CODING RULES (CORE HYPOTHESIS)
<agent_rules>
- **No Math Zone:** DILARANG KERAS melakukan kalkulasi statistik, P-value, t-hitung, atau perkalian matriks di folder ini. Folder ini HANYA mengevaluasi angka yang sudah jadi dari modul sebelumnya.
- **Threshold Reading:** Nilai batas `signifikansi_alpha` WAJIB ditarik dari `pipeline_config.yaml` (default: 0.05). Jangan pernah di-*hardcode*.
- **Anti-Manipulation Constraint:** Jika status H4 berbunyi "Ditolak", skrip ini DILARANG memalsukan atau memaksa *string* menjadi "Diterima". **Pemalsuan data dilarang di tingkat ini.** (Jika pengguna ingin status Diterima, manipulasi harus dilakukan pada Covariance Matrix di tingkat hulu: `core_synthesis/generator_setup.py`).
- **Reporting Output:** Ekspor hasil final dalam format DataFrame rapi ke `reports/tables/ringkasan_hipotesis.xlsx`. Kolom wajib: `Hipotesis`, `Jalur`, `Koefisien`, `P-Value / CI`, `Status`.
</agent_rules>
