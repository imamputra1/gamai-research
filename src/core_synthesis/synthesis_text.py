# src/core_synthesis/08_synthesis_text.py
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from src.utils import load_config, setup_logger

logger = setup_logger("synthesis_text")


def get_kimi_client(config: dict[str, Any]) -> OpenAI:
    """Initialize OpenAI-compatible client for Kimi API from environment.

    Args:
        config: Pipeline configuration dict.

    Returns:
        OpenAI: Configured client instance.

    Raises:
        ValueError: If KIMI_API_KEY environment variable is unset.
    """
    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise ValueError(
            "KIMI_API_KEY environment variable not set. "
            "Export via fish: set -x KIMI_API_KEY 'your-key-here'"
        )

    ai_cfg = config.get("ai", {})
    base_url = ai_cfg.get("kimi_base_url", "https://api.moonshot.cn/v1")

    return OpenAI(api_key=api_key, base_url=base_url)


def load_text_seed(path: Path) -> pd.DataFrame:
    """Load original text responses seed.

    Args:
        path: Path to df_teks.csv.

    Returns:
        pd.DataFrame: Text seed dataframe.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Text seed not found: {path}")
    df: pd.DataFrame = pd.read_csv(path)
    logger.info(f"Text seed loaded: {df.shape}")
    return df


def build_prompt(question_context: str, seed_answers: list[str]) -> tuple[str, str]:
    """Construct system and user prompts for batch qualitative synthesis.

    Args:
        question_context: Column name / survey question text.
        seed_answers: List of original seed responses.

    Returns:
        tuple[str, str]: (system_prompt, user_prompt).
    """
    system_prompt = (
        "Anda adalah asisten peneliti kualitatif senior yang ahli dalam "
        "parafrase dan sintesis narasi survei pasien klinik kesehatan. "
        "Tugas Anda menghasilkan variasi jawaban yang natural, relevan, "
        "dan tidak menyimpang dari konteks medis."
    )

    filtered = [str(a).strip() for a in seed_answers if pd.notna(a) and str(a).strip()]
    seed_text = "\n".join(f"- {ans}" for ans in filtered)

    user_prompt = (
        f"Pertanyaan survei: {question_context}\n\n"
        f"Bank jawaban asli (seed):\n{seed_text}\n\n"
        f"Tugas: Hasilkan tepat 100 variasi jawaban singkat yang merupakan "
        f"parafrase dari bank jawaban di atas. Pertahankan variasi sentimen "
        f"(positif, netral, negatif) sesuai proporsi seed. "
        f"Panjang maksimal 2 kalimat per jawaban. Gunakan bahasa Indonesia natural. "
        f"Output WAJIB dalam format JSON Array murni tanpa markdown: "
        f'["jawaban 1", "jawaban 2", ..., "jawaban 100"]'
    )

    return system_prompt, user_prompt


def call_kimi_batch(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> list[str]:
    """Execute single batch API call to Kimi and parse JSON array response.

    Args:
        client: OpenAI-compatible client.
        model: Model identifier.
        system_prompt: System role prompt.
        user_prompt: User content prompt.
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature.

    Returns:
        list[str]: Parsed list of generated answers. Empty on failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content: str = response.choices[0].message.content or "[]"
        answers = _parse_json_array(content)
        logger.info(f"API returned {len(answers)} answers")
        return answers
    except Exception as exc:
        logger.error(f"API call failed: {exc}")
        return []


def _parse_json_array(content: str) -> list[str]:
    """Extract JSON string array from raw API response text.

    Args:
        content: Raw text response.

    Returns:
        list[str]: List of string items.
    """
    # Direct parse
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(item) for item in data if item is not None]
    except json.JSONDecodeError:
        pass

    # Extract array via regex
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(item) for item in data if item is not None]
        except json.JSONDecodeError:
            pass

    return []


def _resample_from_seed(seed_answers: list[str], target_count: int, rng: random.Random) -> list[str]:
    """Fallback resampler: random pick from seed answers.

    Args:
        seed_answers: Original seed responses.
        target_count: Desired output count.
        rng: Seeded random generator.

    Returns:
        list[str]: Resampled answers.
    """
    pool = [str(a).strip() for a in seed_answers if pd.notna(a) and str(a).strip()]
    if not pool:
        pool = ["Pelayanan sudah memadai."]
    return [rng.choice(pool) for _ in range(target_count)]


def normalize_answers(
    answers: list[str],
    target_count: int,
    seed_answers: list[str],
    rng: random.Random,
) -> list[str]:
    """Ensure exactly target_count answers by truncating or resampling.

    Args:
        answers: Generated answers from API.
        target_count: Required count.
        seed_answers: Original seed for fallback fill.
        rng: Seeded random generator.

    Returns:
        list[str]: Normalized list of length target_count.
    """
    if len(answers) == target_count:
        return answers

    if len(answers) > target_count:
        return answers[:target_count]

    # Fill deficit with resampled seed
    pool = answers + [str(a).strip() for a in seed_answers if pd.notna(a) and str(a).strip()]
    if not pool:
        pool = ["Pelayanan sudah memadai."]

    result = answers.copy()
    while len(result) < target_count:
        result.append(rng.choice(pool))

    return result[:target_count]


def generate_text_dataset(
    df_seed: pd.DataFrame,
    client: OpenAI,
    model: str,
    n_target: int,
    m: int,
    base_seed: int,
) -> pd.DataFrame:
    """Generate one synthetic text dataset via AI batching per column.

    Args:
        df_seed: Original text seed dataframe.
        client: Kimi API client.
        model: Model identifier.
        n_target: Number of synthetic rows.
        m: Current dataset iteration.
        base_seed: Base random seed.

    Returns:
        pd.DataFrame: Synthetic text dataframe (shape: n_target, k).
    """
    rng = random.Random(base_seed + m)
    df_syn: pd.DataFrame = pd.DataFrame()

    for col in df_seed.columns:
        seed_answers = df_seed[col].tolist()

        system_prompt, user_prompt = build_prompt(col, seed_answers)

        answers = call_kimi_batch(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if not answers:
            logger.warning(f"API empty for col '{col}', falling back to seed resampling")
            answers = _resample_from_seed(seed_answers, n_target, rng)

        normalized = normalize_answers(answers, n_target, seed_answers, rng)
        df_syn[col] = normalized

        logger.info(f"Column '{col}' normalized: {len(normalized)} answers")

    return df_syn


def validate_text_shape(df: pd.DataFrame, expected_rows: int, expected_cols: int) -> None:
    """Validate synthetic text dataframe dimensions.

    Args:
        df: Synthetic text dataframe.
        expected_rows: Required row count.
        expected_cols: Required column count.

    Raises:
        ValueError: If shape mismatch.
    """
    if df.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Text synthetic shape harus ({expected_rows}, {expected_cols}). "
            f"Ditemukan: {df.shape}"
        )


def save_text_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Export synthetic text dataframe to CSV without index.

    Args:
        df: Dataframe to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_synthesis_text(
    processed_dir: Path = Path("data/02_processed"),
    synthetic_dir: Path = Path("data/03_synthetic"),
) -> None:
    """Execute full qualitative text synthesis pipeline.

    Args:
        processed_dir: Directory containing df_teks.csv.
        synthetic_dir: Directory for df_teks_syn_*.csv output.
    """
    config: dict[str, Any] = load_config()
    pipeline_cfg = config.get("pipeline", {})
    ai_cfg = config.get("ai", {})

    n_target: int = pipeline_cfg.get("n_target", 100)
    m_datasets: int = pipeline_cfg.get("m_datasets", 20)
    base_seed: int = pipeline_cfg.get("random_seed", 42)
    model: str = ai_cfg.get("kimi_model", "kimi-k2-5")

    client = get_kimi_client(config)

    text_path = processed_dir / "df_teks.csv"
    df_seed = load_text_seed(text_path)

    for m in range(1, m_datasets + 1):
        logger.info(f"Generating text dataset {m}/{m_datasets}")

        df_syn = generate_text_dataset(
            df_seed=df_seed,
            client=client,
            model=model,
            n_target=n_target,
            m=m,
            base_seed=base_seed,
        )

        validate_text_shape(df_syn, expected_rows=n_target, expected_cols=df_seed.shape[1])

        output_path = synthetic_dir / f"df_teks_syn_{m}.csv"
        save_text_dataset(df_syn, output_path)

    logger.info(
        f"Successfully generated {m_datasets} text datasets "
        f"({n_target}x{df_seed.shape[1]})"
    )
    logger.info("Qualitative text synthesis completed successfully.")


if __name__ == "__main__":
    run_synthesis_text()
