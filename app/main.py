from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from .models import solve_nitrification, normalize_params

app = FastAPI(
    title="Nitrification Dynamics Simulator",
    description="Monod + Inhibition ODE model for nitrification kinetics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    S_init: float = Field(default=50, ge=0, le=1000, description="Initial NH4 (mg/L)")
    X_init: float = Field(default=0.1, ge=0, le=10, description="Initial biomass (g/L)")
    time_days: float = Field(default=10, ge=0.1, le=365, description="Simulation duration (days)")
    mu_max: float = Field(default=0.8, ge=0.01, le=5.0, description="Max growth rate (/day)")
    Ks: float = Field(default=2.0, ge=0.1, le=100, description="Half-saturation (mg/L)")
    Y: float = Field(default=0.15, ge=0.01, le=1.0, description="Yield coefficient")
    inhibition_type: str = Field(default="none", pattern="^(none|competitive|uncompetitive|non-competitive)$")
    inhibitor: float = Field(default=0, ge=0, le=1000, description="Inhibitor concentration (mg/L)")
    KI: float = Field(default=100, ge=0.1, le=1000, description="Inhibition constant (mg/L)")


class SimulationResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
async def root():
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return HTMLResponse(content=static_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Nitrification Dynamics Simulator</h1><p>Frontend not found.</p>")


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "Nitrification Dynamics Simulator"}


@app.post("/api/predict", response_model=SimulationResponse)
async def predict(req: SimulationRequest):
    try:
        params = normalize_params({
            "mu_max": req.mu_max,
            "Ks": req.Ks,
            "Y": req.Y,
            "inhibitor": req.inhibitor,
            "KI": req.KI,
            "inhibition_type": req.inhibition_type,
        })

        y0 = [req.S_init, req.X_init]
        t_span = (0, req.time_days)
        t_eval = np.linspace(0, req.time_days, 200)

        sol = solve_nitrification(y0, params, t_span, t_eval)

        return SimulationResponse(
            success=True,
            message="Simulation completed",
            data={
                "time": sol.t.tolist(),
                "S": [max(0, v) for v in sol.y[0].tolist()],
                "X": [max(0, v) for v in sol.y[1].tolist()],
                "params": params,
                "final_S": max(0, sol.y[0][-1]),
                "final_X": max(0, sol.y[1][-1]),
                "removal_pct": round((1 - max(0, sol.y[0][-1]) / req.S_init) * 100, 2) if req.S_init > 0 else 0,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        return SimulationResponse(
            success=False,
            message="Simulation failed",
            error=str(e),
        )
