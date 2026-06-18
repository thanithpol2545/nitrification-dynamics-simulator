# Nitrification Dynamics Simulator

Interactive web application for simulating nitrification kinetics using Monod + Inhibition ODE models.

## Features

- Monod kinetics for NH₄⁺ oxidation
- Competitive, uncompetitive, and non-competitive inhibition models
- Interactive Plotly.js visualization
- Summary metrics (final concentration, removal %)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| ODE Solver | SciPy `solve_ivp` |
| Frontend | HTML + Plotly.js |
| Deployment | Vercel / GitHub |

## Project Structure

```
nitrification-dynamics-simulator/
├── app/
│   ├── main.py              # FastAPI application
│   └── models/
│       ├── base.py          # ODE model definitions
│       └── inhibition.py    # Inhibition factor logic
├── api/
│   └── index.py             # Vercel serverless entry point
├── static/
│   └── index.html           # Interactive frontend
├── vercel.json              # Vercel deployment config
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload

# Open in browser
open http://localhost:8000
```

## API

**POST** `/api/predict`

```json
{
  "S_init": 50,
  "X_init": 0.1,
  "time_days": 10,
  "mu_max": 0.8,
  "Ks": 2.0,
  "Y": 0.15,
  "inhibition_type": "none",
  "inhibitor": 0,
  "KI": 100
}
```

## Deploy to Vercel

```bash
npm i -g vercel
vercel --prod
```

## Model

- **Monod**: μ = μ_max · S / (K_s + S)
- **Competitive**: μ = μ_max · S / (K_s · (1 + I/K_I) + S)
- **Uncompetitive**: μ = μ_max · S / ((1 + I/K_I) · (K_s + S))
- **Non-competitive**: μ = μ_max · S / (K_s + S · (1 + I/K_I))
