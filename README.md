# Nitrification Dynamics Simulator

Interactive web application for simulating **nitrification kinetics** using Monod + Inhibition ODE models, with temperature and pH effects.

**Live demo:** [nitrification-dynamics-simulator.vercel.app](https://nitrification-dynamics-simulator.vercel.app)

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
uvicorn backend.main:app --reload
```

Open http://localhost:8000

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI API endpoints
│   └── models/
│       ├── base.py          # ODE model + temp/pH corrections
│       ├── inhibition.py    # Inhibition factor logic
│       ├── estimation.py    # Parameter estimation
│       └── design.py        # Reactor design calculations
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
| POST | `/api/predict` | Monod simulation |
| POST | `/api/predict/aob-nob` | AOB/NOB two-step simulation |
| POST | `/api/sensitivity` | Parameter sweep analysis |
| POST | `/api/kinetic-analysis` | Kinetic model fitting |
| POST | `/api/parameter-estimation` | Parameter estimation from CSV |
| POST | `/api/design-reactor` | Reactor sizing calculation |
| POST | `/api/oxygen-demand` | Oxygen demand calculation |
| POST | `/api/reactor-comparison` | Batch vs CSTR comparison |

## Deploy

```bash
npm i -g vercel
vercel --prod
```

## References

1. **Monod, J.** (1949). The growth of bacterial cultures. *Annual Review of Microbiology*, 3(1), 371–394. — *Monod kinetics foundation.*

2. **Haldane, J. B. S.** (1930). Enzymes. *Longmans, Green & Co.* — *Substrate inhibition model (Haldane).*

3. **Knowles, G., Downing, A. L., & Barrett, M. J.** (1965). Determination of kinetic constants for nitrifying bacteria in mixed culture with the aid of an electronic computer. *Journal of General Microbiology*, 38(2), 263–278. — *Classic nitrification kinetics parameters.*

4. **Anthonisen, A. C., Loehr, R. C., Prakasam, T. B. S., & Srinath, E. G.** (1976). Inhibition of nitrification by ammonia and nitrous acid. *Journal of the Water Pollution Control Federation*, 48(5), 835–852. — *Inhibition of AOB by free ammonia.*

5. **Wiesmann, U.** (1994). Biological nitrogen removal from wastewater. *Advances in Biochemical Engineering/Biotechnology*, 51, 113–154. — *Comprehensive review of nitrification/denitrification modeling.*

6. **Henze, M., Gujer, W., Mino, T., & van Loosdrecht, M. C. M.** (2000). Activated Sludge Models ASM1, ASM2, ASM2d and ASM3. *IWA Publishing.* — *Industry-standard activated sludge models.*

7. **Ratkowsky, D. A., Lowry, R. K., McMeekin, T. A., Stokes, A. N., & Chandler, R. E.** (1983). Model for bacterial culture growth rate throughout the entire biokinetic temperature range. *Journal of Bacteriology*, 154(3), 1222–1226. — *Temperature correction (Arrhenius-type) for microbial growth.*
