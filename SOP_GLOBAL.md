# SYSTEM_SOP: THESIS_SYNTH_PIPELINE
> **Status:** High-Performance Modular Architecture  
> **Target:** S2 Thesis Synthesis & Analysis (Likert Scale)

## 1. ARCHITECTURE & DIRECTORY MAP
<directory_structure>
thesis_synth_pipeline/
├── data/
│   ├── 01_raw_seed/          # File asli (Excel/CSV) dari responden pilot (n=20)
│   ├── 02_processed/         # Dataframe hasil pembersihan (df_demo, df_likert, df_teks)
│   ├── 03_synthetic/         # Output M-dataset hasil Monte Carlo (df_syn_1 ... df_syn_M)
│   └── 04_report_master/     # Single source of truth untuk Bab 4 (df_report_master.csv)
├── src/
│   ├── core_synthesis/       # BLOK SINTESIS (Monte Carlo, Cholesky, OpenRouter)
│   ├── core_analysis/        # BLOK ANALISIS (Regresi, Mediasi Bootstrap, Validitas)
│   ├── utils/                # Shared utilities (Logger, YAML Loader)
│   └── __init__.py           # WAJIB KOSONG (Cegah RuntimeWarning)
├── config/
│   └── pipeline_config.yaml  # Sentralisasi parameter (Alpha, Seed, Model AI)
├── logs/                     # Audit trail eksekusi (Cek jika API Error/Rate Limit)
├── reports/
│   ├── figures/              # Plot distribusi & grafik mediasi
│   └── tables/               # Output Excel (xlsx) untuk lampiran thesis
├── tests/                    # Unit testing untuk validasi dimensi data
├── .env                      # Kredensial rahasia (OPENROUTER_API_KEY)
└── main.py                   # Global Orchestrator (CLI entry point)
</directory_structure>

## 2. COMPONENT RESPONSIBILITIES
<component_definition>
- **data/**: Immuntable seed data & versioned synthetic output.
- **logs/**: Traceability log per fase. Jika eksekusi berhenti, cek log file terbaru di sini.
- **reports/**: Final output. Tidak boleh ada script yang menulis langsung ke folder root selain ke folder ini.
- **tests/**: Memastikan n_target dan m_datasets sesuai sebelum melakukan regresi berat.
</component_definition>

## 3. AI AGENT CODING SOP (VIBE CODING RULES)
<agent_rules>
- **Persona**: Elite Software Engineer. Code must be DRY, PEP8, and Type-Hinted.
- **No-Init Pattern**: `__init__.py` di setiap sub-package WAJIB kosong. Jangan melakukan import di dalam init untuk menghindari `Double Identity Warning`.
- **Orchestrator Pattern**: Gunakan `orchestrator.py` di setiap package untuk menjalankan workflow internal.
- **Logging vs Printing**: DILARANG menggunakan `print()`. Gunakan `from src.utils import setup_logger`.
- **Config Driven**: Jangan pernah hardcode parameter. Semua angka (N, M, Alpha) dibaca dari `config/pipeline_config.yaml`.
</agent_rules>

## 4. SYNTHESIS DOMAIN LOGIC (BLOK SINTESIS)
<synthesis_rules>
- **Correlation Injection**: Manipulasi matriks kovarians dilakukan di `generator_setup.py` menggunakan target korelasi dari YAML.
- **Likert Integrity**: Data kontinu wajib di-discretize menggunakan `np.round()` dan `np.clip(1, 5)` di `post_processing_likert.py`.
- **Text Synthesis (OpenRouter Fallback)**:
  1. WAJIB 3-Lapis Fallback: OpenRouter.
  2. WAJIB `extra_headers` (HTTP-Referer & Title) untuk menghindari Error 403/404.
  3. WAJIB `time.sleep(5)` di antara request untuk mencegah Rate Limit (429).
  4. Ultimate Fallback: Jika semua AI gagal, gunakan `Seed Resampling`.
</synthesis_rules>

## 5. ANALYSIS DOMAIN LOGIC (BLOK ANALISIS)
<analysis_rules>
- **Mediasi Bootstrap**: Gunakan 5000 iterasi. Pastikan interval kepercayaan (CI) tidak melewati angka 0 untuk status "Diterima".
- **Handover Data**: Blok Analisis membaca data HANYA dari `data/04_report_master/df_report_master.csv`.
</analysis_rules>

## 6. EXECUTION PIPELINE
<workflow>
Fase C (Sintesis): `uv run python -m src.core_synthesis.orchestrator`
Fase D (Analisis): `uv run python main.py --fase 12.1 --fase 12.2 --fase 13.2`
</workflow>
