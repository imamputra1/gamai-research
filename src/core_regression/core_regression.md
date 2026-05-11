# DOMAIN_RULES: src/core_regression/
> **Status:** Mesin Kalkulasi Jalur Langsung (*Direct Effect*)
> **Tujuan:** Menghitung koefisien regresi OLS, t-statistik, dan p-value untuk H1, H2, dan H3.

## 1. FILE ROLES
<file_roles>

| Nama File | Tanggung Jawab Mutlak |
| :--- | :--- |
| `substruktur1_core.py` | Regresi OLS: Independen (X1, X2, X3) $\rightarrow$ Mediator (M). |
| `substruktur2_core.py` | Regresi OLS: Independen (X1, X2, X3) + Mediator (M) $\rightarrow$ Dependen (Y). |
| `orchestrator.py` | Mengeksekusi Substruktur 1 & 2 secara berurutan dan menggabungkan hasil *summary*. |
| `__init__.py` | **WAJIB KOSONG.** |

</file_roles>

## 2. AGENT CODING RULES (CORE REGRESSION)
<agent_rules>
- **Strict Data Source:** Pembacaan data WAJIB dari `data/04_report_master/df_report_master.csv`.
- **Framework OLS:** Gunakan `statsmodels.api` (`sm.OLS`) untuk mendapatkan *summary* komprehensif (R-squared, F-stat, t-stat, p-value). JANGAN gunakan *library* *machine learning* seperti `sklearn` karena tidak mengeluarkan P-Value standar ekonometrika.
- **Constant Injection:** WAJIB menambahkan konstanta (`sm.add_constant(X)`) sebelum melakukan fitting model OLS.
- **Dynamic Variables:** Kolom X, M, dan Y WAJIB dibaca dari blok `arsitektur_model` di `pipeline_config.yaml`. JANGAN *hardcode* nama kolom ("People", "Process", dll) di dalam skrip.
</agent_rules>
