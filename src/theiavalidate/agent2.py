"""Qwen (Hugging Face router) backend -- TESTING ONLY.

A throwaway sibling of agent.py

    HF_TOKEN=hf_... python -m theiavalidate.agent2   # runs the demo below

"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

# Reuse everything that isn't provider-specific.
from theiavalidate.agent import (
    CUSTOM_TOOLS,
    AgentState,
    ToolCall,
    Toolbox,
)

if TYPE_CHECKING:
    from theiavalidate.results import ComparisonResult

# The user-supplied HF router target.
HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B:featherless-ai"

SYSTEM_PROMPT = """\
You answer questions about the result of a TheiaValidate comparison of two tabular
pipeline outputs (a dev table vs. a main/reference table), joined on a key column.

The comparison is already done and deterministic -- do not re-run or second-guess it.
Explain, in plain language, what differs and why it matters to a bioinformatician
reviewing a pipeline change.

Tools:
  - compare_results: the authoritative difference counts and the actual differing
    rows. Start here.
  - get_difference_criteria: the comparison rule for a column (method, python type,
    threshold), so you can say whether a diff is within tolerance or a real mismatch.

Ground every claim in a tool result -- do not invent column names, thresholds, or
values. Investigate the columns that actually differ; do not narrate columns that
matched. Finish with a concise verdict: overall pass/fail, which columns differ, and
whether each looks like noise-within-threshold or a regression.\
"""


def _to_openai_tools(custom_tools: list[dict]) -> list[dict]:
    """Translate our Anthropic-style tool schemas to OpenAI function schemas.

    Anthropic uses {name, description, input_schema}; OpenAI wraps the same JSON
    Schema as {"type": "function", "function": {name, description, parameters}}.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in custom_tools
    ]


def run_agent(
    question: str,
    result: "ComparisonResult",
    *,
    criteria_yaml: Optional[str] = None,
    workflow_name: str = "",
    model: str = DEFAULT_MODEL,
    max_steps: int = 12,
    client: Any = None,
) -> AgentState:
    """Run the agent loop against the HF router and return the populated AgentState.

    Mirrors theiavalidate.agent.run_agent, but the request/response protocol is
    OpenAI chat.completions rather than Anthropic Messages.
    """
    # Lazy import so importing this module doesn't require openai.
    from openai import OpenAI

    box = Toolbox(result, criteria_yaml=criteria_yaml, workflow_name=workflow_name)
    if client is None:
        client = OpenAI(base_url=HF_BASE_URL, api_key=os.environ["HF_TOKEN"])

    tools = _to_openai_tools(CUSTOM_TOOLS)
    state = AgentState(question=question)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": box.initial_prompt(question)},
    ]

    response = None
    while state.steps < max_steps:
        state.steps += 1
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        if response.usage is not None:
            state.input_tokens_used += response.usage.prompt_tokens

        msg = response.choices[0].message

        # Re-append the assistant turn, rebuilt as a plain dict so provider-specific
        # extras (e.g. reasoning_content) don't leak back into the next request.
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in msg.tool_calls
            ]
        messages.append(assistant)

        if not msg.tool_calls:
            break

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                content = box.dispatch(name, args)
            except Exception as exc:  # a tool bug shouldn't kill the run
                content = f"tool error: {exc}"

            state.tool_history.append(ToolCall(call.id, name, args, content))
            state.evidence.append(f"{name}({args})")
            # OpenAI tool results are their own role, keyed by tool_call_id.
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )
    else:
        state.stop_reason = "max_steps"

    if response is not None:
        state.answer = response.choices[0].message.content or ""
        state.stop_reason = state.stop_reason or response.choices[0].finish_reason
    return state


def _demo() -> None:
    """Smoke-test against the theiacov_fasta fixtures. Run from the repo root."""
    from pathlib import Path

    import pandas as pd

    from theiavalidate.config import Config
    from theiavalidate.validator import compare_tables

    root = Path(__file__).resolve().parents[2]
    preset = root / "src/theiavalidate/presets/theiacov_fasta.yaml"
    t1 = root / "tests/phb/theiacov/fasta/theiacov_fasta_v4-1-0.tsv"
    t2 = root / "tests/phb/theiacov/fasta/theiacov_fasta_v4-2-0.tsv"

    def read(p: Path) -> "pd.DataFrame":
        return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)

    config = Config.from_yaml(str(preset)).with_keys(
        key1="entity:theiacov_fasta_v4-1-0_id",
        key2="entity:theiacov_fasta_v4-2-0_id",
    )
    result = compare_tables(read(t1), read(t2), config, left_name="v4-1-0", right_name="v4-2-0")

    state = run_agent(
        "Summarize the differences between the two tables.",
        result,
        criteria_yaml=str(preset),
        workflow_name="theiacov_fasta",
    )
    print("=== answer ===")
    print(state.answer)
    print("\nsteps:", state.steps, "| tokens:", state.input_tokens_used)
    print("tools:", [(tc.tool_name, tc.arguments) for tc in state.tool_history])
    print("stop_reason:", state.stop_reason)


if __name__ == "__main__":
    _demo()
