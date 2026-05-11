# DOMAIN_RULES: src/ai_integration/
> **Status:** Helper Sub-Package (API Client Factory)
> **Tujuan:** Mengisolasi logika koneksi eksternal (LLM API) dari logika bisnis tesis.

## 1. FILE ROLES
<file_roles>

| Nama File | Tanggung Jawab Mutlak |
| :--- | :--- |
| `kimi_client.py` | Menginisiasi *client* OpenAI-compatible khusus untuk Moonshot API (Kimi asli). |
| `openclaw_client.py` | Menginisiasi *client* OpenAI-compatible untuk OpenRouter (Sistem Fallback Gratis). |
| `__init__.py` | **WAJIB KOSONG.** |

</file_roles>

## 2. AGENT CODING RULES (AI INTEGRATION)
<agent_rules>
- **No Prompting Logic:** JANGAN PERNAH meletakkan teks *prompt* (misal: "Buatkan alasan pasien...") di dalam folder ini. *Prompting* adalah wilayah `core_synthesis/synthesis_text.py`.
- **Pure Instantiation:** Folder ini hanya bertugas mengembalikan objek `client = OpenAI(...)`.
- **Header Injection:** Saat membuat *client* OpenRouter, **WAJIB** memasukkan parameter `default_headers` (`HTTP-Referer` dan `X-OpenRouter-Title`) agar lolos dari blokir anti-bot.
- **Environment Driven:** Kunci API (`OPENROUTER_API_KEY`, `KIMI_API_KEY`) WAJIB dibaca menggunakan `os.environ.get()`. JANGAN hardcode API key.
- **Dependency Isolation:** Modul di folder ini DILARANG mengimpor apapun dari folder `core_*`. Mereka harus berdiri secara independen.
</agent_rules>
