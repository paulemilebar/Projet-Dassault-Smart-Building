from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from typing import Iterable, List, Optional, Sequence
from urllib.request import urlopen


@dataclass(frozen=True)
class ContractTariffConfig:
    """Configuration for time-of-use contract tariffs.

    All hours are integers in [0, 23].
    """

    cbuy_peak: float
    cbuy_offpeak: float
    peak_hours: Sequence[int]
    beta_sell: float


@dataclass(frozen=True)
class DayPrices:
    """Predicted prices for a single day."""

    day: date
    cbuy: List[float]
    csell: List[float]


DEFAULT_PEAK_HOURS = tuple(range(8, 20))
OPEN_DPE_API_URL = "https://open-dpe.fr/api/v1/electricity.php"


@dataclass(frozen=True)
class OpenDpeConfig:
    """Configuration for Open DPE tariff fetching."""

    tariff: str = "EDF_bleu"
    option: str = "HC/HP"
    beta_sell: float = 0.6
    hc_hours_weekday: Sequence[int] = tuple(range(9, 18))
    hc_hours_weekend: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class HourlyPrices:
    """Predicted prices for a rolling horizon (e.g., next 24h)."""

    start: datetime
    timestamps: List[datetime]
    cbuy: List[float]
    csell: List[float]
    source_date_tarif: Optional[str] = None


def _validate_hours(hours: Iterable[int]) -> None:
    for h in hours:
        if not isinstance(h, int) or h < 0 or h > 23:
            raise ValueError("Hours must be integers in [0, 23].")


def predict_prices_contract(day: date, config: ContractTariffConfig) -> DayPrices:
    """Predict prices using a fixed time-of-use contract.
    """

    _validate_hours(config.peak_hours)
    peak_set = set(config.peak_hours)

    cbuy = []
    csell = []
    for h in range(24):
        price = config.cbuy_peak if h in peak_set else config.cbuy_offpeak
        cbuy.append(price)
        csell.append(config.beta_sell * price)

    return DayPrices(day=day, cbuy=cbuy, csell=csell)


class ApiPredictionError(RuntimeError):
    pass


def predict_prices_api(day: date, *, fetcher) -> DayPrices:
    """Predict prices by calling an external API via a provided fetcher.

    The fetcher must be a callable: fetcher(day: date) -> Sequence[float]
    It returns a 24-length sequence for Cbuy(t).
    """

    cbuy = fetcher(day)
    if cbuy is None:
        raise ApiPredictionError("Fetcher returned None.")
    if len(cbuy) != 24:
        raise ApiPredictionError("Fetcher must return 24 hourly values for Cbuy.")

    return DayPrices(day=day, cbuy=list(cbuy), csell=[])


def fetch_open_dpe_tariff(
    *,
    tariff: str = "EDF_bleu",
    option: str = "HC/HP",
    timeout_s: int = 10,
) -> tuple[float, float, Optional[str]]:
    """Fetch tariff prices from Open DPE.
    Returns (cbuy_hc, cbuy_hp, date_tarif). For option 'base', hc==hp.
    """

    url = f"{OPEN_DPE_API_URL}?tarif={tariff}"
    with urlopen(url, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    options = payload.get("options", {})
    date_tarif = payload.get("date_tarif")

    if option in ("HC/HP", "heures_creuses", "hc_hp"):
        # API currently uses "heures_creuses" as the key name.
        prices = options.get("heures_creuses", {}).get("prix_kWh", {})
        if not prices:
            prices = options.get("HC/HP", {}).get("prix_kWh", {})
        # API keys can vary in case; normalize.
        if isinstance(prices, dict):
            normalized = {str(k).strip().lower(): v for k, v in prices.items()}
        else:
            normalized = {}
        if "hc" not in normalized or "hp" not in normalized:
            raise ApiPredictionError(
                "Open DPE response missing HC/HP prices. "
                f"Available keys: {list(prices.keys()) if isinstance(prices, dict) else 'N/A'}"
            )
        cbuy_hc = float(normalized["hc"])
        cbuy_hp = float(normalized["hp"])
        return cbuy_hc, cbuy_hp, date_tarif

    if option == "base":
        price = float(options["base"]["prix_kWh"])
        return price, price, date_tarif

    raise ApiPredictionError("Unsupported option. Use 'HC/HP' or 'base'.")


def predict_next_24h_open_dpe(
    *,
    now: Optional[datetime] = None,
    config: OpenDpeConfig = OpenDpeConfig(),
) -> HourlyPrices:
    """Predict prices for the next 24 hours using Open DPE tariffs.
    """

    _validate_hours(config.hc_hours_weekday)
    if config.hc_hours_weekend is not None:
        _validate_hours(config.hc_hours_weekend)

    cbuy_hc, cbuy_hp, date_tarif = fetch_open_dpe_tariff(
        tariff=config.tariff,
        option=config.option,
    )

    start = now or datetime.now()
    hc_weekday = set(config.hc_hours_weekday)
    hc_weekend = (
        set(config.hc_hours_weekend)
        if config.hc_hours_weekend is not None
        else hc_weekday
    )

    timestamps: List[datetime] = []
    cbuy: List[float] = []
    csell: List[float] = []
    for i in range(24):
        ts = start + timedelta(hours=i)
        is_weekday = ts.weekday() < 5
        is_hc = ts.hour in (hc_weekday if is_weekday else hc_weekend)
        price = cbuy_hc if is_hc else cbuy_hp
        timestamps.append(ts)
        cbuy.append(price)
        csell.append(config.beta_sell * price)

    return HourlyPrices(
        start=start,
        timestamps=timestamps,
        cbuy=cbuy,
        csell=csell,
        source_date_tarif=date_tarif,
    )


# Convenience entry point

def predict_prices(
    day: date,
    *,
    mode: str = "contract",
    contract_config: Optional[ContractTariffConfig] = None,
    fetcher=None,
) -> DayPrices:
    """Predict Cbuy(t) and Csell(t) for a given day.

    Modes:
    - contract: fixed peak/off-peak prices
    - api: use external API via fetcher
    """

    if mode == "contract":
        if contract_config is None:
            contract_config = ContractTariffConfig(
                cbuy_peak=0.20,
                cbuy_offpeak=0.12,
                peak_hours=DEFAULT_PEAK_HOURS,
                beta_sell=1.0,
            )
        return predict_prices_contract(day, contract_config)

    if mode == "api":
        if fetcher is None:
            raise ValueError("fetcher is required when mode='api'.")
        return predict_prices_api(day, fetcher=fetcher)

    raise ValueError("Unknown mode. Use 'contract' or 'api'.")


## FOR TESTING

if __name__ == "__main__":
    cfg = OpenDpeConfig(
        tariff="EDF_bleu",
        option="HC/HP",
        beta_sell=0.6,
        hc_hours_weekday=range(9, 18),
    )

    res = predict_next_24h_open_dpe(config=cfg)
    print(res.cbuy)
    print(res.csell)

