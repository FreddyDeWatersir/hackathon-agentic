"""LLM agent template — copy this and improve the prompt to build your agent.

Uses LiteLLM so you can swap models easily via the AGENT_MODEL env var.
This template sends the raw observation to the LLM with a minimal prompt.
It works, but there's a LOT of room to improve:
  - Write a better system prompt with domain strategy
  - Add conversation history so the LLM remembers previous days
  - Filter/summarize the observation to focus on what matters
  - Tune temperature, model choice, etc.
"""

from __future__ import annotations

import json
import os
import sys

import litellm

from agents.runner import run_game

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")

SYSTEM_PROMPT = """\
You manage an Italian restaurant for 30 simulated days. Each day you receive
an observation (JSON) describing your restaurant's state: cash, inventory,
suppliers, menu, reputation, yesterday's service results, and more.

Respond with ONLY a JSON array of tool calls. No explanation, no markdown.

Available tools:
- place_order: {"tool": "place_order", "args": {"supplier": "...", "ingredient": "...", "quantity_kg": N}}
- set_staff_level: {"tool": "set_staff_level", "args": {"level": N}}  (range: 3-15)
- set_price: {"tool": "set_price", "args": {"dish": "...", "price": N}}  (0.8x-1.2x base)
- set_menu: {"tool": "set_menu", "args": {"dishes": [...]}}  (min 5 dishes)
- set_marketing_spend: {"tool": "set_marketing_spend", "args": {"amount": N}}  (0-500 EUR)
- run_happy_hour: {"tool": "run_happy_hour", "args": {}}
- offer_daily_special: {"tool": "offer_daily_special", "args": {"dish": "..."}}
- save_notes: {"tool": "save_notes", "args": {"text": "..."}}  (up to 4000 chars, persists)

Your score = net_profit - penalties (satisfaction, reputation, walkouts, waste).
Going bankrupt (cash < 0) = -100,000 score. Survival is priority #1.

Use the exact supplier, ingredient, and dish names from the observation."""


def strategy(observation: dict, day: int) -> list[dict]:
    user_msg = f"Day {day}/30. Here is today's observation:\n\n{json.dumps(observation, indent=2)}"

    try:
        response = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        tool_calls = json.loads(content)
        if not isinstance(tool_calls, list):
            return []
        return tool_calls

    except Exception as e:
        print(f"  LLM error on day {day}: {e}")
        return []


if __name__ == "__main__":
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY first.")
        print(f"Using model: {MODEL} (override with AGENT_MODEL env var)")
        sys.exit(1)
    print(f"Using model: {MODEL}")
    result = run_game(strategy, team_name="lets_try", seed=42)
