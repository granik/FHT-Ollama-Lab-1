"""Command-line entry point for the student agent."""

import argparse
import json
import os
from pathlib import Path

from .agent import run_agent
from .model import OllamaModel
from .runtime import Runtime, TOOL_SPECS


TRACE_PATH = Path(__file__).resolve().parents[1] / "agent_trace.jsonl"


def build_parser():
    parser = argparse.ArgumentParser(description="Run the Week 1 coding agent")
    parser.add_argument("--root", required=True, help="target workspace")
    parser.add_argument(
        "--mode",
        choices=("read-only", "edit"),
        default="read-only",
    )
    parser.add_argument("--tools", help="comma-separated enabled tools")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--offline-fixture",
        choices=("decimal", "web"),
        help="use a deterministic offline fixture for fetch_url",
    )
    parser.add_argument("--max-turns", type=int, choices=range(1, 16), default=15)
    parser.add_argument("--task", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    enabled_tools = None
    if args.tools:
        enabled_tools = {name.strip() for name in args.tools.split(",") if name.strip()}
        unknown = enabled_tools.difference(TOOL_SPECS)
        if unknown:
            parser.error(f"unknown tools: {', '.join(sorted(unknown))}")

    steps = {}

    def emit(event):
        with TRACE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if event.get("event") == "request":
            try:
                action = json.loads(event["request"])
                label = action.get("tool") or "final"
                arguments = action.get("args", {})
                if isinstance(arguments, dict) and isinstance(arguments.get("path"), str):
                    location = f"path={arguments['path']}"
                elif isinstance(arguments, dict) and isinstance(arguments.get("url"), str):
                    location = f"url={arguments['url']}"
                else:
                    location = f"root={runtime.root}"
            except (json.JSONDecodeError, AttributeError, TypeError):
                label = "invalid JSON"
                location = f"root={runtime.root}"
            steps[event["turn"]] = (label, location)
            print(f"Turn {event['turn']} start: {label} [{location}]")
        elif event.get("event") == "result":
            label, location = steps.get(
                event["turn"], ("unknown", f"root={runtime.root}")
            )
            result = event.get("result", {})
            status = result.get("status") or result.get("termination") or "result"
            print(f"Turn {event['turn']} end: {label} [{location}] -> {status}")

    def approve_bash(command, cwd):
        try:
            answer = input("Approve this command only? Type yes: ")
        except EOFError:
            answer = ""
        approved = answer == "yes"
        emit(
            {
                "event": "approval",
                "command": command,
                "cwd": cwd,
                "approved": approved,
            }
        )
        return approved

    offline_mode = args.offline_fixture or ("decimal" if args.offline else False)
    runtime = Runtime(
        root=args.root,
        mode=args.mode,
        enabled=enabled_tools,
        approve=approve_bash,
        offline=offline_mode,
    )

    model = OllamaModel(model_name=args.model)
    emit(
        {
            "event": "run_start",
            "model": args.model,
            "mode": args.mode,
            "tools": sorted(runtime.available_tool_names()),
        }
    )

    result = run_agent(model, runtime, args.task, max_turns=args.max_turns, emit=emit)
    if result.get("final"):
        print(f"\n{result['final']}")
    if result.get("error"):
        print(f"\nError: {result['error']}")
    print(f"\nAgent terminated with: {result['termination']}")


if __name__ == "__main__":
    main()
