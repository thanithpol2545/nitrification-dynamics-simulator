# Nitrification Dynamics Simulator

Interactive web application for simulating **nitrification kinetics** using Monod + Inhibition ODE models, with temperature and pH effects.

**Live demo:** [nitrification-dynamics-simulator-ra.vercel.app](https://nitrification-dynamics-simulator-ra.vercel.app/)

## Features

- Monod kinetics for NH₄⁺ oxidation with multiple inhibition models
- Temperature correction (Arrhenius) and pH effect
- Interactive Plotly.js visualization with real-time parameter adjustment
- Sensitivity analysis — vary any parameter and see the impact
- Save, load, and compare simulation runs (localStorage)
- CSV export of simulation results

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| ODE Solver | SciPy `solve_ivp` (RK45) |
| Frontend | HTML + Plotly.js |
| Deployment | Vercel |

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI API endpoints
│   └── models/
│       ├── base.py          # ODE model + temp/pH corrections
│       └── inhibition.py    # Inhibition factor logic
├── api/
│   └── index.py             # Vercel serverless entry point
├── static/
│   └── index.html           # Interactive frontend
├── vercel.json
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Run nitrification simulation |
| POST | `/api/sensitivity` | Parameter sweep analysis |
| GET | `/api/export` | Download results as CSV |

## Deploy

```bash
npm i -g vercel
vercel --prod
```
