from __future__ import annotations
import argparse
import sys
import webbrowser
from datetime import date
from pathlib import Path


def _bootstrap_paths() -> tuple[Path, Path]:
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent, cwd.parent.parent]
    project_root = next(
        candidate for candidate in candidates if (candidate / "energy_planner" / "src").exists()
    )
    src_root = project_root / "energy_planner" / "src"

    for candidate in (project_root, src_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return project_root, src_root

PROJECT_ROOT, SRC_ROOT = _bootstrap_paths()

try:
    from energy_planner.src.reporting.dispatch_visualization_parameters import OPENROUTER_API_KEY
except ImportError:
    OPENROUTER_API_KEY = None

from ingestion.load_predicted_inputs import load_predicted_inputs
from main import _run_optimizer
from reporting.optimization_summary import (
    build_optimization_summary_payload,
    try_generate_llm_summary,
)
from state.load_state import load_current_state
from visualization import (
    build_visualization_frame,
    create_dispatch_dashboard,
    save_dashboard_report_html,
    summarize_dispatch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the dispatch dashboard and LLM summary as a standalone HTML report."
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=date.today().isoformat(),
        help="Date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openrouter/free",
        help="LLM model name.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=OPENROUTER_API_KEY,
        help="OpenRouter/OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default=None,
        help="Optional output HTML path.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated HTML in the browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_date = run_date = date(2026, 3, 11)

    if not args.llm_api_key or args.llm_api_key == "PUT_YOUR_OPENROUTER_API_KEY_HERE":
        raise RuntimeError(
            "Missing OpenRouter API key. Set it in utils/dispatch_visualization_parameters.py or pass --llm-api-key."
        )

    predicted_inputs = load_predicted_inputs(run_date=run_date)
    state = load_current_state(run_date=run_date)
    plan_df = _run_optimizer(predicted_inputs, state)

    if plan_df is None:
        raise RuntimeError("Optimizer plan could not be created. Check the CPLEX installation.")

    viz_df = build_visualization_frame(
        predicted_inputs,
        plan_df,
        initial_battery_kwh=state["E_bat_0"],
        battery_capacity_kwh=state["E_max"],
    )
    dispatch_metrics = summarize_dispatch(viz_df)
    print("Dispatch metrics:")
    print(dispatch_metrics)
    
    summary_payload = build_optimization_summary_payload(
        predicted_inputs,
        plan_df,
        initial_battery_kwh=state["E_bat_0"],
        battery_capacity_kwh=state["E_max"],
    )
    summary_text, summary_source = try_generate_llm_summary(
        summary_payload,
        model=args.llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
    )

    fig = create_dispatch_dashboard(
        viz_df,
        title=f"Smart Building Dispatch Dashboard | {run_date.isoformat()}",
    )

    output_path = (
        Path(args.output_html)
        if args.output_html
        else PROJECT_ROOT
        / "energy_planner"
        / "data"
        / "processed"
        / f"dispatch_dashboard_report_{run_date.isoformat()}.html"
    )

    saved = save_dashboard_report_html(
        fig,
        output_path,
        summary_text=summary_text,
        summary_source=summary_source,
        model_name=args.llm_model,
        title=f"Dispatch Dashboard Report | {run_date.isoformat()}",
    )
    print(f"Saved report: {saved}")
    print("\nSummary:\n")
    print(summary_text)

    if not args.no_open:
        webbrowser.open(saved.resolve().as_uri())


if __name__ == "__main__":
    main()
