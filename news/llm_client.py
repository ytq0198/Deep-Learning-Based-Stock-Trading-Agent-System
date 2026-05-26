from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


def create_llm_prompt_fn(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
):
    """Return a callable(prompt) -> json string for LLMEventExtractor."""
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        return None

    endpoint = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model_name = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"

    def _call(prompt: str) -> str:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是金融公告结构化助手。只输出合法 JSON，不要 markdown，不要解释。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        request = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP {err.code}: {detail}") from err

        content = body["choices"][0]["message"]["content"]
        return _extract_json_text(content)

    return _call


def _extract_json_text(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
