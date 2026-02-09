# Projet-Dassault-Smart-Building

Projet 3DS - Optimisation et agents IA pour la gestion de l'energie dans les smart buildings.

## MVP Data Simulator (current status)

The simulator now generates 2 daily datasets (24 hourly rows each):

- `donnees_predites`: forecast-style values (`Tout`, `Tin`, `G`, `PV`, `Pfixe_predit`, `Pflex_predit`, prices).
- `donnees_reelles`: simulated realized values (`Tout_reel`, `Tin_reel`, `PV_reel`, `Pin`, `Pgo`, `Ebat`, `S`, prices).

Primary key used in both tables:

- `(heure, jour, mois, annee)`

Run data generation from repository root:

```bash
python energy_planner/src/main.py --run-date 2026-02-08 --seed 42
```

If `--run-date` is omitted, today's date is used.

The generated files are the following :

- `energy_planner/data/raw/donnees_predites_YYYY-MM-DD.csv`
- `energy_planner/data/raw/donnees_reelles_YYYY-MM-DD.csv`
- `energy_planner/data/processed/donnees_predites_clean_YYYY-MM-DD.csv`
- `energy_planner/data/processed/donnees_reelles_clean_YYYY-MM-DD.csv`

## Brief summary:

- Generate daily predicted/real data.
- Load optimizer inputs from predicted CSV.
- Load system state from `energy_planner/config/parameters.yaml`.
- Run MILP optimizer and print the 24h plan.

Command (from repository root):

```bash
python energy_planner/src/main.py --run-date 2026-02-08 --seed 42
```
