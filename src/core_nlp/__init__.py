# src/core_nlp/__init__.py
from src.core_nlp.orchestrator import NLPPreprocessorOrchestrator
from src.core_nlp.serializer import (
    load_processed,
    load_slang_dict,
    load_stopwords,
    save_processed,
    save_slang_dict,
    save_stopwords,
)
from src.core_nlp.cleaner import clean_text
from src.core_nlp.normalizer import normalize_slang
from src.core_nlp.stopword_filter import remove_stopwords
from src.core_nlp.stemmer import stem_text

__all__ = [
    "NLPPreprocessorOrchestrator",
    "clean_text",
    "normalize_slang",
    "remove_stopwords",
    "stem_text",
    "save_slang_dict",
    "load_slang_dict",
    "save_stopwords",
    "load_stopwords",
    "save_processed",
    "load_processed",
]
