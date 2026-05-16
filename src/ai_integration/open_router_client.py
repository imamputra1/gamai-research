# src/ai_integration/open_router_client.py
import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

from src.ai_integration.constants import DEFAULT_LLM_MODEL, OPENROUTER_BASE_URL

class InsufficientBalanceError(Exception):
    pass

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = OPENROUTER_BASE_URL):
        # Paksa baca .env di sini agar aman
        load_dotenv() 
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        # Pembersihan karakter nakal dari file .env
        if self.api_key:
            self.api_key = self.api_key.strip().replace('"', '').replace("'", "")

        if not self.api_key:
            raise ValueError("API Key OpenRouter tidak ditemukan di environment/file .env.")

        # Inisialisasi SDK OpenAI dengan parameter base_url yang dinamis
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )

    def generate(self, prompt: str, model: str = DEFAULT_LLM_MODEL) -> str:
        """Mengirim prompt ke OpenRouter dan mengembalikan teks."""
        print(f"[INFO] Menghubungi OpenRouter API (Model: {model})...")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            # Ambil konten balasan secara aman
            result = response.choices[0].message.content
            return result if result is not None else ""
        except Exception as e:
            print(f"[ERROR] LLM Generation failed: {e}")
            raise e
