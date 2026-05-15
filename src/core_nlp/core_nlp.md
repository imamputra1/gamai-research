&lt;!-- src/core_nlp/core_nlp.md --&gt;
# src/core_nlp — NLP Text Preprocessing Module

## Arsitektur (SoC)

Modul ini memisahkan 3 concern utama:

1. **Business Logic** (Functional Core)
   - `cleaner.py`      : RegEx-based cleaning & case folding
   - `normalizer.py`   : Dictionary-based slang/typo mapping
   - `stopword_filter.py` : Tokenization & set-intersection filtering
   - `stemmer.py`      : Sastrawi morphological affix stripping

2. **Data Manifestation** (Orchestration)
   - `orchestrator.py` : `NLPPreprocessorOrchestrator` mengkomposisikan
     4 tahap fungsional menjadi pipeline DataFrame-aware.

3. **Serialization Format**
   - `serializer.py`   : JSON persistence untuk slang dict / stopwords;
     CSV persistence untuk DataFrame hasil preprocessing.

## Pipeline (4 Tahap)

| Tahap | Fungsi | Output Kolom |
|-------|--------|--------------|
| 15.1.1 | `clean_text()` | `text_cleaned` |
| 15.1.2 | `normalize_slang()` | `text_normalized` |
| 15.1.3 | `remove_stopwords()` | `text_filtered` |
| 15.1.4 | `stem_text()` | `text_final_preprocessed` |

## Penggunaan

```python
from src.core_nlp import NLPPreprocessorOrchestrator

orch = NLPPreprocessorOrchestrator(
    custom_stopwords=["rumah", "sakit", "pasien"]
)
df_processed = orch.run(df_raw, text_column="jawaban_responden")
