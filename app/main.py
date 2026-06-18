import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .models import (
    solve_nitrification,
    normalize_params,
    sensitivity_analysis,
    compute_effective_mu_max,
)

app = FastAPI(
    title="Nitrification Dynamics Simulator",
    description="Monod + Inhibition ODE model for nitrification kinetics with temperature and pH effects",
    version="2.0.0",
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
    mu_max: float = Field(default=0.8, ge=0.01, le=5.0, description="Max growth rate at 20°C (/day)")
    Ks: float = Field(default=2.0, ge=0.1, le=100, description="Half-saturation (mg/L)")
    Y: float = Field(default=0.15, ge=0.01, le=1.0, description="Yield coefficient")
    inhibition_type: str = Field(default="none", pattern="^(none|competitive|uncompetitive|non-competitive)$")
    inhibitor: float = Field(default=0, ge=0, le=1000, description="Inhibitor concentration (mg/L)")
    KI: float = Field(default=100, ge=0.1, le=1000, description="Inhibition constant (mg/L)")
    temperature: float = Field(default=20, ge=0, le=50, description="Temperature (°C)")
    pH: float = Field(default=7.5, ge=0, le=14, description="pH value")


class SimulationResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
    error: str | None = None


class SensitivityRequest(BaseModel):
    base_params: dict
    param_name: str
    values: list[float]
    S_init: float = 50
    X_init: float = 0.1
    time_days: float = 10


@app.get("/", response_class=HTMLResponse)
async def root():
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return HTMLResponse(content=static_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Nitrification Dynamics Simulator</h1><p>Frontend not found.</p>")


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "Nitrification Dynamics Simulator v2"}


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
            "temperature": req.temperature,
            "pH": req.pH,
        })

        y0 = [req.S_init, req.X_init]
        t_span = (0, req.time_days)
        t_eval = np.linspace(0, req.time_days, 200)

        sol = solve_nitrification(y0, params, t_span, t_eval)
        mu_eff = compute_effective_mu_max(params)

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
                "effective_mu_max": round(mu_eff, 4),
                "temp_factor": round(params["mu_max"] * (1.07 ** (params["temperature"] - 20)) / max(params["mu_max"], 1e-10), 4),
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


class SensitivityResponse(BaseModel):
    success: bool
    data: list[dict] | None = None
    error: str | None = None


@app.post("/api/sensitivity", response_model=SensitivityResponse)
async def sensitivity(req: SensitivityRequest):
    try:
        base_params = normalize_params(req.base_params.copy())
        init_cond = [req.S_init, req.X_init]
        t_span = (0, req.time_days)
        t_eval = np.linspace(0, req.time_days, 100)

        results = sensitivity_analysis(base_params, req.param_name, req.values, init_cond, t_span, t_eval)

        return SensitivityResponse(success=True, data=results)

    except Exception as e:
        return SensitivityResponse(success=False, error=str(e))


@app.get("/api/export")
async def export_csv(
    S_init: float = Query(50),
    X_init: float = Query(0.1),
    time_days: float = Query(10),
    mu_max: float = Query(0.8),
    Ks: float = Query(2.0),
    Y: float = Query(0.15),
    inhibition_type: str = Query("none"),
    inhibitor: float = Query(0),
    KI: float = Query(100),
    temperature: float = Query(20),
    pH: float = Query(7.5),
):
    try:
        params = normalize_params({
            "mu_max": mu_max, "Ks": Ks, "Y": Y,
            "inhibitor": inhibitor, "KI": KI,
            "inhibition_type": inhibition_type,
            "temperature": temperature, "pH": pH,
        })

        t_span = (0, time_days)
        t_eval = np.linspace(0, time_days, 200)
        sol = solve_nitrification([S_init, X_init], params, t_span, t_eval)

        lines = ["time_days,NH4_mgL,Biomass_gL"]
        for t, s, x in zip(sol.t, sol.y[0], sol.y[1]):
            lines.append(f"{t:.4f},{max(0,s):.4f},{max(0,x):.4f}")

        return PlainTextResponse(
            content="\n".join(lines),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=nitrification_simulation.csv"},
        )

    except Exception as e:
        return PlainTextResponse(content=f"Error: {str(e)}", status_code=400)
