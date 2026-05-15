# src/core_nlp/serializer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

import pandas as pd


def save_slang_dict(path: str, slang_dict: Dict[str, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slang_dict, f, ensure_ascii=False, indent=2)


def load_slang_dict(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, str] = json.load(f)
    return data


def save_stopwords(path: str, stopwords: Set[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(stopwords), f, ensure_ascii=False, indent=2)


def load_stopwords(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data: list[str] = json.load(f)
    return set(data)


def save_processed(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_processed(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")
