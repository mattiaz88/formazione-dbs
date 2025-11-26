#%% md
# Part 8 — Mini‑Exercises (Groq + Gemini)
#
# This file contains 8 hands‑on exercises to practice prompt design and output control with two providers:
# - Groq (OpenAI‑compatible API, model: `gpt-oss-20b`)
# - Google AI Studio / Gemini (SDK `google-generativeai`, model: `gemini-2.5-flash`)
#
# Notes
# - Set your keys as environment variables before running: `export GROQ_API_KEY=...` and `export GEMINI_API_KEY=...`.
# - Install libs if needed: `pip install openai google-generativeai python-dotenv`.
# - Try each exercise on both providers when possible.

#%%
# Optional installs (uncomment if needed)
# !pip install openai google-generativeai python-dotenv

#%% md
# Setup — Clients and helper functions
#
# - `call_groq(prompt, ...)`
# - `call_gemini(prompt, ...)`
# - `call_llm(prompt, provider='groq'|'gemini', ...)`

#%%
import os
import json
from typing import Any, Optional

# Groq uses the OpenAI‑compatible SDK
from openai import OpenAI
import google.generativeai as genai

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GROQ_API_KEY:
    print("Note: GROQ_API_KEY is not set. Set it to call Groq.")
if not GEMINI_API_KEY:
    print("Note: GEMINI_API_KEY is not set. Set it to call Gemini.")

_groq_client: Any = None
_gemini_ready: bool = False

if GROQ_API_KEY:
    _groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_ready = True


def call_groq(prompt: str,
              temperature: float = 0.0,
              max_tokens: int = 800,
              model: str = "gpt-oss-20b") -> str:
    """Call Groq's OpenAI-compatible chat API."""
    if _groq_client is None:
        return "Groq client not configured. Set GROQ_API_KEY and re-run."
    try:
        resp = _groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Groq call failed: {e}"


def call_gemini(prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 800,
                model: str = "gemini-2.5-flash") -> str:
    """Call Google Gemini via google-generativeai SDK."""
    if not _gemini_ready:
        return "Gemini not configured. Set GEMINI_API_KEY and re-run."
    try:
        model_instance = genai.GenerativeModel(model)
        gen_cfg = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = model_instance.generate_content(prompt, generation_config=gen_cfg)
        return (response.text or "").strip()
    except Exception as e:
        return f"Gemini call failed: {e}"


def call_llm(prompt: str,
             provider: str = "groq",
             temperature: float = 0.0,
             max_tokens: int = 800,
             model: Optional[str] = None,
             verbose: bool = True) -> str:
    """Unified caller for 'groq' or 'gemini'."""
    p = provider.lower().strip()
    if p == "groq":
        model = model or "gpt-oss-20b"
        if verbose:
            print(f"[Groq] model={model} T={temperature} max_tokens={max_tokens}")
        return call_groq(prompt, temperature=temperature, max_tokens=max_tokens, model=model)
    if p == "gemini":
        model = model or "gemini-2.5-flash"
        if verbose:
            print(f"[Gemini] model={model} T={temperature} max_tokens={max_tokens}")
        return call_gemini(prompt, temperature=temperature, max_tokens=max_tokens, model=model)
    return "Unknown provider. Use 'groq' or 'gemini'."

#%% md
# 1) Temperature sweep
# Generate 5 headline variants for a sustainability campaign; compare T=0.0 vs T=1.0.
#
# - Try both providers.
# - Observe determinism vs creativity.

#%%
headline_prompt = (
    "Generate 5 short, punchy headline variants for a sustainability campaign about cutting plastic waste.\n"
    "Constraints: 6–9 words each; upbeat; no emojis."
)

print("\n== Groq — T=0.0 ==")
print(call_llm(headline_prompt, provider="groq", temperature=0.0, max_tokens=200, verbose=False))

print("\n== Groq — T=1.0 ==")
print(call_llm(headline_prompt, provider="groq", temperature=1.0, max_tokens=200, verbose=False))

print("\n== Gemini — T=0.0 ==")
print(call_llm(headline_prompt, provider="gemini", temperature=0.0, max_tokens=200, verbose=False))

print("\n== Gemini — T=1.0 ==")
print(call_llm(headline_prompt, provider="gemini", temperature=1.0, max_tokens=200, verbose=False))

#%% md
# 2) Format control
# Ask for a 3‑row Markdown table with columns: `Metric | Definition | Why it matters`.
#
# - Verify the table renders and has exactly 3 rows of data.

#%%
table_prompt = (
    "Produce a Markdown table with columns: Metric | Definition | Why it matters.\n"
    "Include exactly 3 data rows relevant to TV/CTV campaign analytics."
)
print(call_llm(table_prompt, provider="groq", temperature=0.2, max_tokens=200, verbose=False))
print("\n---\n")
print(call_llm(table_prompt, provider="gemini", temperature=0.2, max_tokens=200, verbose=False))

#%% md
# 3) Style mimic
# Provide one example claim and ask the model to mimic the style for a new product.
#
# - Keep outputs concise and benefit‑led.

#%%
style_prompt = (
    "Rewrite in the style of the example.\n\n"
    "Example:\nInput: 'Our battery lasts 20 hours so you can keep going all day.'\n"
    "Output: 'Power through your day with a 20-hour battery.'\n\n"
    "Now you try:\nInput: 'Our platform lets marketers analyze TV campaigns faster.'\nOutput:"
)
print(call_llm(style_prompt, provider="groq", temperature=0.3, max_tokens=120, verbose=False))
print("\n---\n")
print(call_llm(style_prompt, provider="gemini", temperature=0.3, max_tokens=120, verbose=False))

#%% md
# 4) Guarded claims
# Give budget and reach goals; ask the model to propose 3 tactics and include 1 risk per tactic.
#
# - Ensure each tactic contains a short risk note.

#%%
brief = (
    "Budget: €500k. Goal: A25-54 reach ≥ 55% with frequency ≥ 3 using TV + CTV.\n"
    "Task: Propose 3 concise tactics, each with one associated risk note."
)
print(call_llm(brief, provider="groq", temperature=0.2, max_tokens=250, verbose=False))
print("\n---\n")
print(call_llm(brief, provider="gemini", temperature=0.2, max_tokens=250, verbose=False))

#%% md
# 5) Chain‑of‑Thought (CoT)
# Ask for a step‑by‑step short reasoning to classify performance as `good`/`ok`/`poor`.
#
# - First try without CoT, then with a short reasoning.

#%%
no_cot = (
    "Classify performance as good, ok, or poor: 'Reach 52%, frequency 2.1 vs target 55%/≥3'.\n"
    "Answer with one word."
)
with_cot = (
    "Classify performance as good, ok, or poor: 'Reach 52%, frequency 2.1 vs target 55%/≥3'.\n"
    "Explain reasoning briefly (2-3 steps), then give the final label."
)
print("No-CoT:\n", call_llm(no_cot, provider="groq", temperature=0.0, verbose=False))
print("\nWith CoT:\n", call_llm(with_cot, provider="groq", temperature=0.0, verbose=False))

#%% md
# 6) JSON output + validation
# Request a specific JSON schema and validate keys in Python; if invalid, ask the model to fix.
#
# - Schema: `{ "channel": str, "reach_target": int (40–80), "risk_note": str }`
# - TODO: Implement `validate_payload(data)`.

#%%
json_prompt = (
    "Return a JSON object with keys 'channel', 'reach_target', and 'risk_note'.\n"
    "Constraints: 'reach_target' is an integer between 40 and 80. Use double quotes only and no extra text."
)
raw = call_llm(json_prompt, provider="groq", temperature=0.1, max_tokens=120, verbose=False)
print("Raw model output:\n", raw)

# TODO: Implement validation
# def validate_payload(data: dict) -> tuple[bool, str]:
#     """Return (is_valid, error_message_if_any)."""
#     pass

# After implementing, uncomment the lines below to test the repair loop.
# try:
#     obj = json.loads(raw)
# except Exception as e:
#     print("JSON parse error:", e)
# else:
#     ok, err = validate_payload(obj)
#     if ok:
#         print("Valid JSON ✅", obj)
#     else:
#         print("Invalid JSON ❌:", err)
#         fix_prompt = (
#             "The following JSON is invalid based on the schema. Fix it.\n"
#             f"Schema: channel:str, reach_target:int(40-80), risk_note:str\n"
#             f"JSON: {raw}\n"
#             "Return JSON only."
#         )
#         fixed = call_llm(fix_prompt, provider="gemini", temperature=0.0, max_tokens=120, verbose=False)
#         print("\nFixed attempt:\n", fixed)

#%% md
# 7) Prompt repair
# Give a poor prompt; ask the model to rewrite it to be precise and testable.
#
# - Consider specifying role, task, constraints, and format.

#%%
poor_prompt = "write something about ads"
repair_request = (
    "Rewrite the following poor prompt to be precise, constrained, and testable.\n"
    "Return the improved prompt only.\n\n"
    f"Poor prompt: {poor_prompt}"
)
print(call_llm(repair_request, provider="groq", temperature=0.2, max_tokens=180, verbose=False))
print("\n---\n")
print(call_llm(repair_request, provider="gemini", temperature=0.2, max_tokens=180, verbose=False))

#%% md
# 8) Provider compare
# Run the same prompt on Groq and Gemini; note differences in style/latency.
#
# - Optional: time the calls using `time.perf_counter()`.

#%%
import time
compare_prompt = (
    "In one sentence, explain a simple strategy to reduce ad fatigue in a month-long TV campaign."
)

start = time.perf_counter()
resp_groq = call_llm(compare_prompt, provider="groq", temperature=0.2, max_tokens=80, verbose=False)
lat_groq = time.perf_counter() - start

start = time.perf_counter()
resp_gem = call_llm(compare_prompt, provider="gemini", temperature=0.2, max_tokens=80, verbose=False)
lat_gem = time.perf_counter() - start

print("Groq (s):", round(lat_groq, 3), "\n", resp_groq)
print("\nGemini (s):", round(lat_gem, 3), "\n", resp_gem)

print("\nNote: Latency varies by network and provider load; run multiple times for a fair comparison.")
