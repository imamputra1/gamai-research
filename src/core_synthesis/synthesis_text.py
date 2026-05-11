# src/core_synthesis/synthesis_text.py
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.utils import load_config, setup_logger

load_dotenv()

logger = setup_logger("synthesis_text")


def get_ai_client(config: dict[str, Any]) -> OpenAI:
    """Initialize OpenAI-compatible client for OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY tidak ditemukan di environment. "
            "Pastikan file .env sudah dibuat di root directory."
        )

    ai_cfg = config.get("ai", {})
    base_url = ai_cfg.get("kimi_base_url", "https://openrouter.ai/api/v1")

    return OpenAI(api_key=api_key, base_url=base_url)


def load_text_seed(path: Path) -> pd.DataFrame:
    """Load original text responses seed."""
    if not path.exists():
        raise FileNotFoundError(f"Text seed not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Text seed loaded: {df.shape}")
    return df


def generate_text_with_fallback(
    prompt_text: str, client: OpenAI, ai_cfg: dict[str, Any]
) -> str:
    """Eksekusi LLM dengan sistem Failover 3 lapis."""
    models_config = ai_cfg.get("models", ["google/gemma-2-9b-it:free"])
    if isinstance(models_config, str):
        models = [models_config]
    else:
        models = models_config

    max_tokens = ai_cfg.get("max_tokens", 4096)
    temperature = ai_cfg.get("temperature", 0.7)

    for model_name in models:
        try:
            logger.debug(f"Mencoba sintesis teks dengan: {model_name}...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Anda adalah asisten medis yang ahli membuat variasi alasan kunjungan pasien klinik dalam bahasa Indonesia yang natural.",
                    },
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://github.com/tesis-s2",
                    "X-OpenRouter-Title": "Sintesis S2",
                },
            )

            logger.info(f"Sukses menggunakan {model_name}")
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"{model_name} GAGAL. Error: {e}. Beralih ke cadangan...")
            continue

    logger.error("FATAL: Semua model AI gagal merespons. Jatuh ke Seed Resampling.")
    return ""


def generate_text_dataset(
    df_seed: pd.DataFrame,
    client: OpenAI,
    ai_cfg: dict[str, Any],
    n_target: int,
    m: int,
    base_seed: int,
) -> pd.DataFrame:
    """Generate synthetic text responses with AI and fallback to resampling."""
    random_seed = base_seed + m
    random.seed(random_seed)

    synthetic_data = {}

    for col in df_seed.columns:
        time.sleep(5)

        prompt = (
            f"Buatkan {n_target} variasi alasan/jawaban pasien yang natural berbahasa Indonesia "
            f"berdasarkan pertanyaan kuesioner klinik berikut:\n'{col}'\n"
            f"Sangat Penting: Berikan HANYA {n_target} baris teks jawaban. "
            f"Jangan gunakan nomor, bulet, atau awalan apapun. Pisahkan setiap jawaban dengan baris baru (enter)."
        )

        result_text = generate_text_with_fallback(prompt, client, ai_cfg)

        answers = []
        if result_text:
            raw_answers = [
                re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
                for line in result_text.split("\n")
                if line.strip()
            ]

            if len(raw_answers) >= n_target:
                answers = raw_answers[:n_target]
            else:
                logger.warning(
                    f"API menghasilkan kurang dari target ({len(raw_answers)}/{n_target}), "
                    f"falling back to seed resampling"
                )
        else:
            logger.warning(
                f"API empty for col '{col}', falling back to seed resampling"
            )

        if len(answers) < n_target:
            answers = (
                df_seed[col]
                .dropna()
                .sample(n=n_target, replace=True, random_state=random_seed)
                .tolist()
            )

        logger.info(f"Column '{col}' normalized: {len(answers)} answers")
        synthetic_data[col] = answers

    return pd.DataFrame(synthetic_data)


def validate_text_shape(
    df: pd.DataFrame, expected_rows: int, expected_cols: int
) -> None:
    """Validate synthetic text dataframe dimensions."""
    if df.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Text synthetic shape harus ({expected_rows}, {expected_cols}). "
            f"Ditemukan: {df.shape}"
        )


def save_text_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Export synthetic text dataframe to CSV without index."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_synthesis_text(
    processed_dir: Path = Path("data/02_processed"),
    synthetic_dir: Path = Path("data/03_synthetic"),
) -> None:
    """Execute full qualitative text synthesis pipeline."""
    config: dict[str, Any] = load_config()
    pipeline_cfg = config.get("pipeline", {})
    ai_cfg = config.get("ai", {})

    n_target: int = pipeline_cfg.get("n_target", 100)
    m_datasets: int = pipeline_cfg.get("m_datasets", 20)
    base_seed: int = pipeline_cfg.get("random_seed", 42)

    client = get_ai_client(config)

    text_path = processed_dir / "df_teks.csv"
    df_seed = load_text_seed(text_path)

    for m in range(1, m_datasets + 1):
        logger.info(f"Generating text dataset {m}/{m_datasets}")

        df_syn = generate_text_dataset(
            df_seed=df_seed,
            client=client,
            ai_cfg=ai_cfg,
            n_target=n_target,
            m=m,
            base_seed=base_seed,
        )

        validate_text_shape(
            df_syn, expected_rows=n_target, expected_cols=df_seed.shape[1]
        )

        output_path = synthetic_dir / f"df_teks_syn_{m}.csv"
        save_text_dataset(df_syn, output_path)

        logger.info(f"Teks sintetik disimpan: {output_path}")


if __name__ == "__main__":
    run_synthesis_text()
