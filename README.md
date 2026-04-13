# Projet-Dassault-Smart-Building
Projet 3DS - Optimisation et agents IA pour la gestion de l'energie dans les smart buildings.
---
## Getting started
### Prerequisites
| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 recommended |
| IBM CPLEX | 22.1 | Only needed to run the optimizer - everything else works without it |
---
### 1. Clone the repository
`ash
git clone https://github.com/paulemilebar/Projet-Dassault-Smart-Building.git
cd Projet-Dassault-Smart-Building
`
---
### 2. Create a virtual environment
**Windows**
`ash
python -m venv .venv
.venv\Scripts\activate
`
**macOS / Linux**
`ash
python3 -m venv .venv
source .venv/bin/activate
`
You should see (.venv) at the start of your terminal prompt.
---
### 3. Install dependencies
`ash
pip install --upgrade pip
pip install -r requirements.txt
`
> **CPLEX note** - cplex is listed in 
equirements.txt but it requires IBM CPLEX to be installed on your machine first.
> If you do not have it, simply ignore the install error: the pipeline detects its absence and skips the optimizer step automatically.
> Free academic licenses are available at [ibm.com/academic](https://www.ibm.com/academic).
---
### 4. Optional LLM setup
The dashboard summary uses the openai Python SDK against an OpenAI-compatible endpoint.
The current default setup in the project is OpenRouter with:
- model: openrouter/free
- base URL: https://openrouter.ai/api/v1
You have two ways to provide credentials.
**Option A - local config file used by the dashboard scripts**
Create energy_planner/src/reporting/dispatch_visualization_parameters.py with:
`python
OPENROUTER_API_KEY = "your_openrouter_key_here"
`
This file is local-only and must not be committed.
**Option B - environment variables used by 	ry_generate_llm_summary(...)**
**Windows**
`ash
set OPENAI_API_KEY=your_key_here
set OPENAI_MODEL=openrouter/free
set OPENAI_BASE_URL=https://openrouter.ai/api/v1
`
**macOS / Linux**
`ash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=openrouter/free
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
`
Important nuance: the helper function 	ry_generate_llm_summary(...) supports a deterministic fallback when the package, key, model, or API call is unavailable, but the standalone dashboard scripts still check for an API key before running. In practice, if you want to generate the dashboard with LLM text from the provided scripts, you should configure a real key.
---
### 5. Train the demand model (first run only)
The simulator depends on a pre-trained demand model. Generate and train it once before running the full pipeline:
`ash
python -m Predictor_demand.train_user_demand_model \
  --train-end-date 2026-03-11 \
  --num-train-days 90 \
  --num-valid-days 10 \
  --output-dataset energy_planner/data/processed/synthetic_user_demand_history.csv
`
This writes the model bundle to Predictor_demand/models/user_demand_rf_bundle.joblib.
---
### 6. Run the main pipeline
`ash
python energy_planner/src/main.py --run-date 2026-03-19 --seed 42
`
This command:
- predicts the day inputs
- simulates the realized day
- writes the raw and processed CSV files
- loads the optimizer inputs and state
- runs the MILP optimizer
- updates the online-learning models
Important: energy_planner/src/main.py does **not** generate the dashboard or the LLM summary anymore.
---
## MVP Data Simulator (current status)
The simulator now generates 2 daily datasets (24 hourly rows each):
- donnees_predites: forecast-style values (Tout, Tin, G, PV, Pfixe_predit, Pflex_predit, prices)
- donnees_reelles: simulated realized values (Tout_reel, Tin_reel, PV_reel, Pin, Pgo, Ebat, S, prices)
Primary key used in both tables:
- (heure, jour, mois, annee)
The generated files are:
- energy_planner/data/raw/donnees_predites_YYYY-MM-DD.csv
- energy_planner/data/raw/donnees_reelles_YYYY-MM-DD.csv
- energy_planner/data/processed/donnees_predites_clean_YYYY-MM-DD.csv
- energy_planner/data/processed/donnees_reelles_clean_YYYY-MM-DD.csv
---
## Dashboard And LLM Summary
There are currently two entry points related to the dashboard summary.
### 1. Recommended reporting script
energy_planner/src/reporting/dashboard_24h_with_vizualization.py
This is the latest version. It:
- loads predicted inputs and the optimizer state
- runs the optimizer
- computes a baseline vs optimizer comparison
- builds a structured summary payload
- calls the LLM through 	try_generate_llm_summary(...)
- generates a single HTML report containing both Plotly charts and the natural-language summary
Configuration is done directly in the file through these constants:
- RUN_DATE
- LLM_MODEL
- LLM_BASE_URL
- OUTPUT_HTML
- OPEN_REPORT_IN_BROWSER
Run it from the repository root:
`ash
python energy_planner/src/reporting/dashboard_24h_with_vizualization.py
`
The API key is loaded from:
`python
from energy_planner.src.reporting.dispatch_visualization_parameters import OPENROUTER_API_KEY
`
### 2. utility script
utils/dispatch_visualization.py
This script also builds an HTML dashboard and LLM summary, and it accepts CLI arguments such as:
- --llm-model
- --llm-api-key
- --llm-base-url
- --output-html
- --no-open
Example:
`ash
python utils/dispatch_visualization.py --llm-api-key your_key_here --llm-model openrouter/free --llm-base-url https://openrouter.ai/api/v1
`
Note: the script exposes a --run-date argument, but the current code still forces 
un_date = date(2026, 3, 11) internally. So for now the generated report is tied to that hard-coded date unless the script itself is updated.
---
## How The LLM Summary Works
The dashboard does not send charts or raw HTML to the model.
Instead, the flow is:
1. The optimizer produces a 24h dispatch plan with Pin, Pgo, PV, Pch, Pdis, Ebat, and S.
2. build_visualization_frame(...) and summarize_dispatch(...) derive aggregated dispatch metrics.
3. build_optimization_summary_payload(...) creates a compact payload containing:
   - total served demand
   - grid import/export
   - battery charge/discharge and state of charge
   - PV production and self-consumption
   - flexible demand served/curtailed
   - energy costs and revenues
   - hourly peaks
   - operating-regime breakdown
   - optionally baseline vs optimizer metrics
   - optionally the optimization objective, variables, constraints, and parameters
4. 	try_generate_llm_summary(...) builds a French prompt from that payload.
5. The function first tries client.responses.create(...) from the openai SDK.
6. If that fails, it retries with client.chat.completions.create(...).
7. If both fail, or if the package / credentials / model are missing, it falls back to a deterministic template summary.
So the LLM is used as a natural-language explanation layer on top of structured optimization outputs.

## LLM Summary In The Notebook
The notebook utils/dispatch_visualization_draft.ipynb can also generate a natural-language dispatch summary.
Recommended OpenRouter setup inside the notebook:
`python
llm_base_url = "https://openrouter.ai/api/v1"
llm_model = "openrouter/free"
llm_api_key = "PUT_YOUR_OPENROUTER_API_KEY_HERE"
`
Each user should create and use their own API key locally.
---
## How It Works Globally
The data is currently simulated:
- predicted data is generated by the simulator and prediction modules
- realized data is simulated afterward
Execution order for the main planning pipeline:
1. energy_planner/src/prediction/predict_day.py: generate the predicted day inputs
2. energy_planner/src/simulator/generate_data.py: simulate the realized day and save daily CSVs
3. energy_planner/src/ingestion/load_predicted_inputs.py: load the optimizer-ready predicted inputs
4. energy_planner/src/state/load_state.py: load fixed system settings from YAML
5. energy_planner/src/optimization/optimizer.py: solve the MILP dispatch plan
6. energy_planner/src/main.py: orchestrate the end-to-end daily pipeline
Important distinction:
- predicted inputs = hourly forecast variables such as PV, Pfix, Pflex, and prices
- state/config = fixed constraints and weights such as battery limits, efficiencies, grid limits, and objective coefficients
---
## Demand Forecasting Package
The repository includes a separate demand-forecasting module in Predictor_demand/.
It trains two random-forest models:
- one for fixed demand: Pfixe
- one for flexible demand: Pflex
The feature set is intentionally minimal and interpretable:
- hour
- day
- month
- year
- Tout
- Tin
- occupancy
- heating_gap_outdoor
- elow_tmin_flag
Train the demand models from repository root:
`ash
python -m Predictor_demand.train_user_demand_model --train-end-date 2026-03-11 --num-train-days 90 --num-valid-days 10 --output-dataset energy_planner/data/processed/synthetic_user_demand_history.csv
`
This writes the bundle to:
- Predictor_demand/models/user_demand_rf_bundle.joblib
Training prints validation metrics directly:
- MAE: average absolute prediction error
- RMSE: larger errors are penalized more strongly
- R2: how well the model explains the target variance (1.0 is best)
