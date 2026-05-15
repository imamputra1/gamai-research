# src/core_nlp/preprocessing.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm

from src.core_nlp.constants import (
    COL_TEXT_CLEANED,
    COL_TEXT_FINAL,
    COL_TEXT_FILTERED,
    COL_TEXT_NORMALIZED,
    DEFAULT_SLANG_DICT,
    DEFAULT_STOPWORDS,
)


# =============================================================================
# 15.1.1: Pembersihan Teks Dasar (Cleaning & Case Folding)
# =============================================================================
def clean_text(series: pd.Series, output_column: Optional[str] = None) -> pd.Series:
    """Lowercase, replace punctuation/digits with space, collapse spaces.

    Kritis: Tanda baca berdempetan jadi spasi, bukan langsung hapus.
    Output murni [a-z\s].
    """
    cleaned: pd.Series = series.astype(str).str.lower()
    cleaned = cleaned.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    cleaned = cleaned.str.replace(r"[0-9]", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    result: pd.Series = cleaned.str.strip()
    if output_column:
        result.name = output_column
    return result


# =============================================================================
# 15.1.2: Normalisasi Kata Tidak Baku (Slang/Typo Mapping)
# =============================================================================
def normalize_slang(
    series: pd.Series,
    slang_dict: Optional[Dict[str, str]] = None,
    output_column: Optional[str] = None,
) -> pd.Series:
    """Dictionary-based replacement dengan word boundary \b."""
    mapping: Dict[str, str] = slang_dict if slang_dict is not None else DEFAULT_SLANG_DICT
    result: pd.Series = series.copy()
    for slang, standard in mapping.items():
        pattern: str = r"\b" + re.escape(slang) + r"\b"
        result = result.str.replace(pattern, standard, regex=True)
    if output_column:
        result.name = output_column
    return result


# =============================================================================
# 15.1.3: Stopwords Removal (Penyaringan Kata Hubung)
# =============================================================================
def remove_stopwords(
    series: pd.Series,
    custom_stopwords: Optional[List[str]] = None,
    output_column: Optional[str] = None,
) -> pd.Series:
    """Tokenization & set-intersection filtering."""
    stopwords: Set[str] = DEFAULT_STOPWORDS.copy()
    if custom_stopwords:
        stopwords.update(custom_stopwords)

    def _filter(text: str) -> str:
        words: List[str] = text.split()
        filtered: List[str] = [w for w in words if w not in stopwords]
        return " ".join(filtered)

    result: pd.Series = series.apply(_filter)
    if output_column:
        result.name = output_column
    return result


# =============================================================================
# 15.1.4: Stemming Bahasa Indonesia (Pemotongan Imbuhan)
# =============================================================================
def stem_text(series: pd.Series, output_column: Optional[str] = None) -> pd.Series:
    """Sastrawi stemming dengan tqdm progress tracking.

    Warning: Wajib dijalankan SETELAH stopwords removal (15.1.3).
    """
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed: List[str] = [
        stemmer.stem(text) for text in tqdm(series, desc="Stemming", unit="doc")
    ]
    result: pd.Series = pd.Series(stemmed, index=series.index)
    if output_column:
        result.name = output_column
    return result


# =============================================================================
# Mapping Output (Text -> Stage mapping)
# =============================================================================
def build_text_mapping(df: pd.DataFrame, text_column: str) -> Dict[str, Dict[str, str]]:
    """Build mapping from original text to each preprocessing stage output."""
    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        original: str = str(row[text_column])
        mapping[original] = {
            "cleaned": str(row.get(COL_TEXT_CLEANED, "")),
            "normalized": str(row.get(COL_TEXT_NORMALIZED, "")),
            "filtered": str(row.get(COL_TEXT_FILTERED, "")),
            "final": str(row.get(COL_TEXT_FINAL, "")),
        }
    return mapping


def save_text_mapping(mapping: Dict[str, Dict[str, str]], output_path: str) -> None:
    """Persist text mapping to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_text_mapping(path: str) -> Dict[str, Dict[str, str]]:
    """Load persisted text mapping from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, str]] = json.load(f)
    return data


def export_mapping_csv(df: pd.DataFrame, text_column: str, output_path: str) -> None:
    """Export mapping as flat CSV for human inspection."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "original": df[text_column],
        "cleaned": df.get(COL_TEXT_CLEANED, ""),
        "normalized": df.get(COL_TEXT_NORMALIZED, ""),
        "filtered": df.get(COL_TEXT_FILTERED, ""),
        "final": df.get(COL_TEXT_FINAL, ""),
    })
    out_df.to_csv(output_path, index=False, encoding="utf-8")


def get_unique_tokens(df: pd.DataFrame, stage: str = "final") -> List[str]:
    """Extract unique tokens from a preprocessing stage."""
    col_map = {
        "cleaned": COL_TEXT_CLEANED,
        "normalized": COL_TEXT_NORMALIZED,
        "filtered": COL_TEXT_FILTERED,
        "final": COL_TEXT_FINAL,
    }
    col = col_map.get(stage, COL_TEXT_FINAL)
    all_text = " ".join(df[col].astype(str).replace("nan", "")).split()
    return sorted(list(set(token for token in all_text if token.strip())))


# =============================================================================
# Serialization Format
# =============================================================================
def save_slang_dict(path: str, slang_dict: Dict[str, str]) -> None:
    """Serialize slang dictionary ke JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slang_dict, f, ensure_ascii=False, indent=2)


def load_slang_dict(path: str) -> Dict[str, str]:
    """Deserialize slang dictionary dari JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, str] = json.load(f)
    return data


def save_stopwords(path: str, stopwords: Set[str]) -> None:
    """Serialize stopwords set ke JSON list."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(stopwords), f, ensure_ascii=False, indent=2)


def load_stopwords(path: str) -> Set[str]:
    """Deserialize stopwords set dari JSON list."""
    with open(path, "r", encoding="utf-8") as f:
        data: list[str] = json.load(f)
    return set(data)


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Serialize processed DataFrame ke CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_processed(path: str) -> pd.DataFrame:
    """Deserialize processed DataFrame dari CSV."""
    return pd.read_csv(path, encoding="utf-8")
