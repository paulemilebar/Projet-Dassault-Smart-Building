from datetime import date
from pathlib import Path

from simulator.schema import SimulationConfig


def _parse_scalar(value: str):
    """Parse a scalar value from a simple YAML-like line."""
    text = value.strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        return text


def _read_simple_yaml(path: Path) -> dict:
    """
    Read a flat key:value YAML file.
    This is intentionally simple for MVP and avoids extra dependencies.
    """
    data: dict = {}
    if not path.exists():
        return data

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def load_current_state(
    run_date: date | None = None,
    config_path: str | Path = "energy_planner/config/parameters.yaml",
) -> dict:
    """
    Load the current state used by the optimizer.

    For MVP:
    - base values come from SimulationConfig defaults
    - optional overrides come from config YAML
    """
    cfg = SimulationConfig()
    raw = _read_simple_yaml(Path(config_path))

    # Optimizer-required battery and constraint parameters.
    state = {
        "E_bat_0": float(raw.get("E_bat_0", cfg.ebat_initial_kwh)),
        "E_max": float(raw.get("E_max", cfg.ebat_max_kwh)),
        "P_ch_max": float(raw.get("P_ch_max", cfg.pch_max_kw)),
        "P_dis_max": float(raw.get("P_dis_max", cfg.pdis_max_kw)),
        "eta_ch": float(raw.get("eta_ch", cfg.eta_ch)),
        "eta_dis": float(raw.get("eta_dis", cfg.eta_dis)),
        # User preferences and simple metadata for downstream modules.
        "tfrigo": float(raw.get("tfrigo", cfg.tfrigo_c)),
        "Tmin": float(raw.get("Tmin", cfg.tmin_c)),
        "Tmax": float(raw.get("Tmax", cfg.tmax_c)),
        # Parametres objectif/contraintes reseau pour l'optimiseur.
        "C_L": float(raw.get("C_L", 2.0)),
        "C_bat": float(raw.get("C_bat", 0.005)),
        "C_emissions_grid": float(raw.get("C_emissions_grid", 1.0)),
        "C_emissions_PV": float(raw.get("C_emissions_PV", 0.02)),
        "P_g_max_import": float(raw.get("P_g_max_import", 7.0)),
        "P_g_max_export": float(raw.get("P_g_max_export", 4.0)),
        "run_date": run_date.isoformat() if run_date else None,
    }

    # Minimal safety checks.
    if state["E_bat_0"] < 0 or state["E_bat_0"] > state["E_max"]:
        raise ValueError("Invalid state: E_bat_0 must be in [0, E_max]")
    if state["P_ch_max"] <= 0 or state["P_dis_max"] <= 0:
        raise ValueError("Invalid state: charge/discharge power limits must be > 0")
    if not (0 < state["eta_ch"] <= 1) or not (0 < state["eta_dis"] <= 1):
        raise ValueError("Invalid state: efficiencies must be in (0, 1]")
    if state["P_g_max_import"] < 0 or state["P_g_max_export"] < 0:
        raise ValueError("Invalid state: grid import/export limits must be >= 0")

    return state
