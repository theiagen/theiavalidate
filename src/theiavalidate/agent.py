"""
Dump of potential agent harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import json
import yaml

import anthropic

if TYPE_CHECKING:  # keep heavy/optional imports out of module import
    from theiavalidate.results import ComparisonResult

# Set defualt model, this can be changed.
DEFAULT_MODEL = "claude-opus-4-8"

# The PHB resources the web tools are allowed to touch.
DOCS_DOMAIN = "theiagen.github.io"
REPO_DOMAIN = "github.com/theiagen/public_health_bioinformatics"


@dataclass
class ToolCall:
    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None

@dataclass
class AgentState:
    question: str
    # raw api log of what was executed
    tool_history: list[ToolCall] = field(default_factory=list)
    # facts we want model to read
    evidence: list[str] = field(default_factory=list)
    answer: str | None = None
    stop_reason: str | None = None
    steps: int = 0
    input_tokens_used: int = 0

    @property
    def is_done(self) -> bool:
        return self.answer is not None or self.stop_reason is not None

    def evidence_summary(self) -> str:
        if not self.evidence:
            return "No evidence gathered yet"
        return "\n".join(
            f"[{i+1}] {e}" for i, e in enumerate(self.evidence)
        )

    def last_tool(self) -> str | None:
        return self.tool_history[-1].tool_name if self.tool_history else None


def get_difference_criteria(workflow_name: str, criteria_yaml: str) -> dict:
    """Get the configured yaml for the specific workflow"""

    # read in criteria yaml
    with open(criteria_yaml) as fh:
        loaded_yaml = yaml.safe_load(fh)

    return {"workflow_name": workflow_name, "difference_criteria": loaded_yaml}


def _json(obj: Any) -> str:
    """Tool results must be strings; sets/floats/etc. fall back to str()."""
    return json.dumps(obj, default=str, indent=2)


CUSTOM_TOOLS: list[dict] = [
    {
        "name": "compare_results",
        "description": (
            "Read the deterministic comparison results (dev table vs. reference "
            "table, joined on a key). With no column: the overall verdict, per-column "
            "difference counts, and any rows/columns exclusive to one table. With a "
            "column: the differing rows for that column (left value, right value, and "
            "percent_diff for numeric columns). Start here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "A column to drill into. Omit for the overview.",
                }
            },
        },
    },
    {
        "name": "get_difference_criteria",
        "description": (
            "Read the configured comparison rule from the criteria YAML. With no "
            "column: the join key and every configured column with its method. With a "
            "column: that column's full rule (method, expected python type, threshold, "
            "delimiter/parse, source-column mappings). Use it to decide whether a diff "
            "is within tolerance or a hard mismatch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "A column name. Omit to list all configured columns.",
                }
            },
        },
    },
]


def _web_tools(max_uses: int = 8) -> list[dict]:
    """Server-side web search + fetch, scoped to the PHB docs and repo."""
    domains = [DOCS_DOMAIN, REPO_DOMAIN]
    return [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "allowed_domains": domains,
            "max_uses": max_uses,
        },
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "allowed_domains": domains,
            "max_uses": max_uses,
        },
    ]


SYSTEM_PROMPT = f"""\
You answer questions about the result of a TheiaValidate comparison of two tabular
pipeline outputs (a dev table vs. a main/reference table), joined on a key column.

The comparison is already done and deterministic -- you do not re-run or second-guess
it. Explain, in plain language, what differs and why it likely matters to a
bioinformatician reviewing a pipeline change.

Tools:
  - compare_results: the authoritative difference counts and the actual differing
    rows. Start here.
  - get_difference_criteria: the comparison rule for a column (method, python type,
    threshold), so you can say whether a diff is within tolerance or a real mismatch.
  - web_search / web_fetch: the Theiagen Public Health Bioinformatics docs
    ({DOCS_DOMAIN}) and repo ({REPO_DOMAIN}) -- use to explain what a field means or
    how a task computes it.

Ground every claim in a tool result -- do not invent column names, thresholds, or
values. Investigate the columns that actually differ; do not narrate columns that
matched. Finish with a concise verdict: overall pass/fail, which columns differ,
whether each looks like noise-within-threshold or a regression, and any exclusive
rows/columns.\
"""


class Toolbox:
    """Holds the comparison context and implements the client-side tools.

    Carries the deterministic `ComparisonResult` (for compare_results) and the path
    to the criteria YAML (for get_difference_criteria). `registry` maps a tool name
    to its implementation; `dispatch` routes a tool_use block by name.
    """

    def __init__(
        self,
        result: "ComparisonResult",
        *,
        criteria_yaml: Optional[str] = None,
        workflow_name: str = "",
    ) -> None:
        self.result = result
        self.criteria_yaml = criteria_yaml
        self.workflow_name = workflow_name

        self.registry: dict[str, Callable[[dict], str]] = {
            "compare_results": self.compare_results,
            "get_difference_criteria": self.criteria,
        }

    def dispatch(self, name: str, tool_input: dict) -> str:
        fn = self.registry.get(name)
        if fn is None:
            return f"unknown tool {name!r}"
        return fn(tool_input or {})

    def initial_prompt(self, question: str) -> str:
        """Seed the run with the deterministic overview so the model starts grounded."""
        return (
            f"{question}\n\n"
            "Deterministic overview (counts only) -- inspect the differing columns "
            "with the tools before answering:\n\n" + _json(self.result.to_dict())
        )

    def compare_results(self, args: dict) -> str:
        result = self.result
        column = args.get("column")

        if not column:
            overview = result.to_dict()
            overview["columns_with_differences"] = [
                name for name, col in result.columns.items() if col.n_differences
            ]
            return _json(overview)

        col = result.columns.get(column)
        if col is None:
            return _json(
                {
                    "error": f"no such compared column {column!r}",
                    "available": sorted(result.columns),
                }
            )

        percent = col.percent_diff  # per-row Series or None
        rows = []
        for key in col.left.index:
            row = {result.key: key, "left": col.left.get(key), "right": col.right.get(key)}
            if percent is not None and key in percent.index:
                row["percent_diff"] = percent.get(key)
            rows.append(row)

        return _json(
            {
                "column": column,
                "method": col.method,
                "n_compared": col.n_compared,
                "n_differences": col.n_differences,
                "differing_rows": rows,
            }
        )
    
    def criteria(self, args: dict) -> str:
        if not self.criteria_yaml:
            return _json({"error": "no criteria_yaml was provided to the agent"})

        loaded = get_difference_criteria(self.workflow_name, self.criteria_yaml)
        spec = loaded["difference_criteria"] or {}
        columns = spec.get("columns", {})
        column = args.get("column")

        if not column:
            return _json(
                {
                    "workflow_name": self.workflow_name,
                    "key": spec.get("key") or [spec.get("key1"), spec.get("key2")],
                    "columns": {
                        name: rule.get("method", "any_of")
                        for name, rule in columns.items()
                    },
                }
            )

        rule = columns.get(column)
        if rule is None:
            return _json(
                {
                    "error": f"no criteria configured for {column!r}",
                    "available": sorted(columns),
                }
            )
        return _json({"column": column, "rule": rule})


def run_agent(
    question: str,
    result: "ComparisonResult",
    *,
    criteria_yaml: Optional[str] = None,
    workflow_name: str = "",
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    max_tokens: int = 8000,
    max_steps: int = 12,
    client: Any = None,
) -> AgentState:
    """Run the agent loop and return the populated AgentState.

    The model inspects the deterministic differences, the criteria, and the PHB
    docs/repo through the tool surface, then answers `question`.

    Args:
        question: what to ask about the comparison (e.g. "summarize the diffs").
        result: the deterministic ComparisonResult from validator.py.
        criteria_yaml: path to the criteria YAML (enables get_difference_criteria).
        workflow_name: label passed through to the criteria reader.
        model: Claude model id (defaults to Opus 4.8, adaptive thinking).
        effort: reasoning effort -- low | medium | high | max.
        max_tokens: per-response output cap.
        max_steps: hard cap on agent loop iterations.
        client: an anthropic.Anthropic (or compatible) client; env-created if omitted.
    """

    box = Toolbox(result, criteria_yaml=criteria_yaml, workflow_name=workflow_name)
    if client is None:
        client = anthropic.Anthropic()

    state = AgentState(question=question)
    system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # stable prefix: system + tools
        }
    ]
    tools = CUSTOM_TOOLS + _web_tools()
    messages: list[dict] = [{"role": "user", "content": box.initial_prompt(question)}]

    response = None
    while state.steps < max_steps:
        state.steps += 1
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            system=system,
            tools=tools,
            messages=messages,
        )
        if response.usage is not None:
            state.input_tokens_used += response.usage.input_tokens

        # Server-side web tools hit their internal loop cap: re-send to resume.
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason != "tool_use":
            break

        # Client-side custom tool call(s). Keep the assistant turn, then answer each.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use" or block.name not in box.registry:
                continue  # web_search / web_fetch are executed server-side
            try:
                content = box.dispatch(block.name, block.input)
                is_error = False
            except Exception as exc:  # a tool bug shouldn't kill the run
                content, is_error = f"tool error: {exc}", True

            state.tool_history.append(
                ToolCall(block.id, block.name, dict(block.input), content)
            )
            state.evidence.append(f"{block.name}({dict(block.input)})")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})
    else:
        state.stop_reason = "max_steps"

    if response is not None:
        state.answer = "".join(b.text for b in response.content if b.type == "text")
        state.stop_reason = state.stop_reason or response.stop_reason
    return state
