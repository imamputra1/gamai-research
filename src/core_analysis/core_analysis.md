# DOMAIN_RULES: src/core_analysis/
> **Status:** Evaluator Kualitas Data (Blok C/D)
> **Tujuan:** Uji Asumsi Klasik, Validitas, Reliabilitas, dan Analisis Deskriptif.

## 1. FILE ROLES
<file_roles>

| Nama File | Tanggung Jawab Mutlak |
| :--- | :--- |
| `master_selection.py` | Memilih satu dataset terbaik dari `03_synthetic/` dan memindahkannya ke `04_report_master/df_report_master.csv`. |
| `uji_validitas.py` | Menghitung Pearson Correlation (r-hitung vs r-tabel). Menyimpan hasil ke Excel. |
| `uji_reliabilitas.py` | Menghitung Cronbach's Alpha (> 0.6 atau 0.7 sesuai konfigurasi). |
| `uji_normalitas.py` | Melakukan Uji Kolmogorov-Smirnov atau Shapiro-Wilk pada residual regresi. |
| `uji_multikol.py` | Menghitung VIF (Variance Inflation Factor) dan Tolerance. Batas VIF < 10. |
| `uji_hetero.py` | Melakukan Uji Glejser atau mencetak *Scatterplot* (ZPRED vs SRESID). |
| `deskriptif_likert.py` | Menghitung Mean, Median, Modus, dan frekuensi persentase (TCR) per indikator. |
| `profiling_demografi.py`| Membuat *pie chart* / tabel silang demografi responden. |
| `qc_final_check.py` | Verifikasi akhir sebelum data masuk ke tahap `core_regression`. |
| `orchestrator.py` | Merangkai semua uji di atas secara sekuensial. |
| `__init__.py` | **WAJIB KOSONG.** |

</file_roles>

## 2. AGENT CODING RULES (CORE ANALYSIS)
<agent_rules>
- **Strict Read-Only:** JANGAN PERNAH mengubah, membulatkan, atau memodifikasi nilai `df` di dalam folder ini. Ini murni tahap *Read & Evaluate*. Jika data jelek, perbaikan dilakukan di `core_synthesis`, BUKAN di sini.
- **Single Source of Truth:** Semua modul pengujian (uji_*) WAJIB membaca file dari `data/04_report_master/df_report_master.csv`. Jangan baca dari folder synthetic lagi.
- **Dynamic Thresholds:** Batas kritis seperti `alpha = 0.05` atau batas `VIF = 10` WAJIB dibaca dari file `pipeline_config.yaml`.
- **Export Standards:**
  - Tabel hasil uji (misal status Valid/Tidak Valid) WAJIB diekspor menggunakan fungsi dari `src.utils.excel_exporter`.
  - Simpan output tabel ke `reports/tables/`.
  - Simpan output grafik (Heteroskedastisitas, Normalitas P-Plot) ke `reports/figures/`.
- **Pandas Ecosystem:** Gunakan `statsmodels` atau `scipy.stats` untuk pengujian statistik tingkat lanjut.
</agent_rules>
