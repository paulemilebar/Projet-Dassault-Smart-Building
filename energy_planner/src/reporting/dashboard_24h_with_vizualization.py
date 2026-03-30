from __future__ import annotations

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

from energy_planner.src.reporting.dispatch_visualization_parameters import OPENROUTER_API_KEY

from baseline.calculator import BaselineCalculator
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

RUN_DATE = date(2026, 3, 11)
LLM_MODEL = "openrouter/free"
LLM_BASE_URL = "https://openrouter.ai/api/v1"
OUTPUT_HTML: str | None = None
OPEN_REPORT_IN_BROWSER = True


def main() -> None:
    run_date = RUN_DATE

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "PUT_YOUR_OPENROUTER_API_KEY_HERE":
        raise RuntimeError(
            "Missing OpenRouter API key. Set it in utils/dispatch_visualization_parameters.py."
        )

    predicted_inputs = load_predicted_inputs(run_date=run_date)
    state = load_current_state(run_date=run_date)
    plan_df = _run_optimizer(predicted_inputs, state)

    if plan_df is None:
        raise RuntimeError("Optimizer plan could not be created. Check the CPLEX installation.")

    calculator = BaselineCalculator()
    baseline_metrics = calculator.compute_grid_only(df=predicted_inputs, state=state)
    opt_metrics = calculator.compute_optimizer_performance(
        plan_df=plan_df,
        inputs_df=predicted_inputs,
        state=state,
    )
    real_comp = {
        "base_cost": baseline_metrics["cost"],
        "opt_cost": opt_metrics["cost"],
        "base_co2": baseline_metrics["emissions"],
        "opt_co2": opt_metrics["emissions"],
    }

    viz_df = build_visualization_frame(
        predicted_inputs,
        plan_df,
        initial_battery_kwh=state["E_bat_0"],
        battery_capacity_kwh=state["E_max"],
    )
    dispatch_metrics = summarize_dispatch(viz_df, comparison_results=real_comp)
    print("Dispatch metrics:")
    print(dispatch_metrics)

    summary_payload = build_optimization_summary_payload(
        predicted_inputs,
        plan_df,
        initial_battery_kwh=state["E_bat_0"],
        battery_capacity_kwh=state["E_max"],
        comparison_results=real_comp,
        optimizer_state=state,
    )
    summary_text, summary_source = try_generate_llm_summary(
        summary_payload,
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=LLM_BASE_URL,
    )

    fig = create_dispatch_dashboard(
        viz_df,
        comparison_results=real_comp,
    )

    output_path = (
        Path(OUTPUT_HTML)
        if OUTPUT_HTML
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
        model_name=LLM_MODEL,
        title=f"Smart Building Dashboard Report | {run_date.isoformat()}",
    )
    print(f"Saved report: {saved}")
    print("\nSummary:\n")
    print(summary_text)

    if OPEN_REPORT_IN_BROWSER:
        webbrowser.open(saved.resolve().as_uri())


if __name__ == "__main__":
    main()
