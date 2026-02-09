from __future__ import annotations

import argparse
from datetime import date, datetime

from ingestion.load_predicted_inputs import load_predicted_inputs
from state.load_state import load_current_state
from simulator.generate_data import SimulationConfig, generate_and_save_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP daily smart-building datasets.")
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format. Default: today",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    return parser.parse_args()


def parse_run_date(raw: str | None) -> date:
    if raw is None:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()


def main() -> None:
    args = parse_args()
    run_date = parse_run_date(args.run_date)
    cfg = SimulationConfig(seed=args.seed)
    paths = generate_and_save_day(run_date=run_date, cfg=cfg)
    predicted_inputs = load_predicted_inputs(run_date=run_date)
    state = load_current_state(run_date=run_date)

    print(f"Run date: {run_date.isoformat()}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("\nLoaded predicted inputs (optimizer contract):")
    print(predicted_inputs.head(3).to_string(index=False))
    print(f"... total rows: {len(predicted_inputs)}")
    print("\nLoaded state:")
    print(state)


if __name__ == "__main__":
    main()
