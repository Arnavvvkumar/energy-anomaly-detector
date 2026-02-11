# Energy Forecasting Project

Simple energy consumption forecasting pipeline using household power data.

## Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Put dataset at:
`energy_forecasting_project/data/household_power_consumption.txt`

3. Run main pipeline:
```bash
python energy_forecasting_project/notebooks/energy_forecasting.py
```

4. Optional honest baseline run:
```bash
python energy_forecasting_project/honest_energy_forecasting_results.py
```

## Notes

- Data is split chronologically (train/validation/test).
- Default feature setup is leakage-safe.
- Saved models are written under `models/`.
