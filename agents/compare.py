"""Run all baseline agents and compare scores. """

from __future__ import annotations

import os
import sys

from agents.runner import run_game
from agents.do_nothing import strategy as do_nothing_strategy
from agents.naive_rule import strategy as naive_rule_strategy


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RESTBENCH_URL", "http://localhost:8001")
    seed = 42

    print("=" * 60)
    print("DO-NOTHING AGENT")
    print("=" * 60)
    do_nothing_result = run_game(
        do_nothing_strategy, base_url=base_url, team_name="do_nothing", seed=seed
    )

    print("\n" + "=" * 60)
    print("NAIVE RULE AGENT")
    print("=" * 60)
    naive_result = run_game(
        naive_rule_strategy, base_url=base_url, team_name="naive_rule", seed=seed
    )

    llm_template_result = None
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from agents.llm_template import strategy as llm_template_strategy
            print("\n" + "=" * 60)
            print(f"LLM TEMPLATE AGENT ({os.getenv('AGENT_MODEL', 'openai/gpt-4.1-mini')})")
            print("=" * 60)
            llm_template_result = run_game(
                llm_template_strategy, base_url=base_url, team_name="llm_template", seed=seed
            )
        except ImportError:
            print("\nSkipping LLM agent (llm_template not available)")
    else:
        print("\nSkipping LLM agent (no API key set)")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    rows = [
        ("Do-Nothing", do_nothing_result),
        ("Naive Rule", naive_result),
    ]
    if llm_template_result:
        rows.append(("LLM Template", llm_template_result))

    print(f"{'Agent':<20} {'Score':>10} {'Profit':>10} {'Days':>6} {'Final Cash':>12}")
    print("-" * 62)
    for name, r in rows:
        s = r["score"]
        print(
            f"{name:<20} {s['total_score']:>10.0f} {s['net_profit']:>10.0f} "
            f"{r['days_survived']:>6} {r['final_cash']:>12.0f}"
        )


if __name__ == "__main__":
    main()
