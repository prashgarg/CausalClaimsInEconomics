from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config at {path}: expected a YAML mapping.")
    return cfg


def resolve_path(config_path: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, strict: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                if strict:
                    raise
                continue
    return out


def clean_for_key(text: str) -> str:
    text = str(text or "")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def parse_json_content(content: str) -> Dict[str, Any]:
    if content is None:
        raise ValueError("Empty model content.")
    s = str(content).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def get_openai_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "openai package is required for --execute mode. Install with `pip install openai`."
        ) from e
    return OpenAI()


def run_chat_request(
    client: Any,
    body: Dict[str, Any],
    retries: int = 3,
    sleep_seconds: float = 1.0,
) -> Any:
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return client.chat.completions.create(**body)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(sleep_seconds)
    if last_err is None:
        raise RuntimeError("Unknown request error.")
    raise last_err


def get_message_content(response_obj: Any) -> str:
    try:
        return response_obj.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Unable to parse model response content.") from e


def custom_id_parts(custom_id: str) -> Dict[str, Any]:
    parts = {"paper_id": custom_id, "stage1_iteration": None, "stage2_iteration": None}
    m1 = re.match(r"^(.*?)__s1_i(\d+)$", custom_id)
    if m1:
        parts["paper_id"] = m1.group(1)
        parts["stage1_iteration"] = int(m1.group(2))
        return parts
    m2 = re.match(r"^(.*?)__s1_i(\d+)__s2_i(\d+)$", custom_id)
    if m2:
        parts["paper_id"] = m2.group(1)
        parts["stage1_iteration"] = int(m2.group(2))
        parts["stage2_iteration"] = int(m2.group(3))
        return parts
    return parts
