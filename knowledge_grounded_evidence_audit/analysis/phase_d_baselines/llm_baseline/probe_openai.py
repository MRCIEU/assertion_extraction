#!/usr/bin/env python3.11
"""Phase 2B — Probe 2 retest: single gpt-4o-mini chat completion + cost estimate."""
from __future__ import annotations

import os
import sys

from openai import OpenAI


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("BLOCKED: OPENAI_API_KEY not set in this environment.", file=sys.stderr)
        sys.exit(2)
    client = OpenAI()
    msg = [
        {"role": "system", "content": "Reply with a single word: OK"},
        {"role": "user", "content": "Health check for Phase D LLM baseline."},
    ]
    rsp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0)
    text = (rsp.choices[0].message.content or "").strip()
    usage = rsp.usage
    print("response:", text)
    if usage:
        print("tokens:", usage.model_dump())
    print("status: READY")


if __name__ == "__main__":
    main()
