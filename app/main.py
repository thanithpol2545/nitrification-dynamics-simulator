import csv
import io
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Nitrification Dynamics Simulator", description="Comprehensive nitrification kinetics modeling platform", version="3.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ---- Schemas ----

class SimulationRequest(BaseModel):
    S_init: float = Field(default=50, ge=0, le=1000)
    X_init: float = Field(default=0.1, ge=0, le=10)
    time_days: float = Field(default=10, ge=0.1, le=365)
    mu_max: float = Field(default=0.8, ge=0.01, le=5.0)
    Ks: float = Field(default=2.0, ge=0.1, le=100)
    Y: float = Field(default=0.15, ge=0.01, le=1.0)
    inhibition_type: str = Field(default="none", pattern="^(none|competitive|uncompetitive|non-competitive)$")
    inhibitor: float = Field(default=0, ge=0, le=1000)
    KI: float = Field(default=100, ge=0.1, le=1000)
    temperature: float = Field(default=20, ge=0, le=50)
    pH: float = Field(default=7.5, ge=0, le=14)
    DO: float = Field(default=4.0, ge=0, le=20)
    model_type: str = Field(default="single", pattern="^(single|aob_nob)$")


class AOBNoBRequest(BaseModel):
    NH4_init: float = Field(default=50, ge=0, le=1000)
    NO2_init: float = Field(default=0, ge=0, le=500)
    NO3_init: float = Field(default=0, ge=0, le=500)
    X_AOB_init: float = Field(default=0.1, ge=0, le=10)
    X_NOB_init: float = Field(default=0.05, ge=0, le=10)
    time_days: float = Field(default=10, ge=0.1, le=365)
    mu_max_AOB: float = Field(default=0.8, ge=0.01, le=5.0)
    K_NH4: float = Field(default=2.0, ge=0.1, le=100)
    Y_AOB: float = Field(default=0.15, ge=0.01, le=1.0)
    mu_max_NOB: float = Field(default=1.0, ge=0.01, le=5.0)
    K_NO2: float = Field(default=1.5, ge=0.1, le=100)
    Y_NOB: float = Field(default=0.08, ge=0.01, le=1.0)
    temperature: float = Field(default=20, ge=0, le=50)
    pH: float = Field(default=7.5, ge=0, le=14)
    DO: float = Field(default=4.0, ge=0, le=20)


class DesignRequest(BaseModel):
    S_in: float = Field(default=50, ge=1, le=1000)
    Q: float = Field(default=1000, ge=1, le=1e6, description="Flow rate (m3/day)")
    target_S_eff: float = Field(default=5, ge=0.1, le=500)
    mu_max: float = Field(default=0.8, ge=0.01, le=5.0)
    Ks: float = Field(default=2.0, ge=0.1, le=100)
    Y: float = Field(default=0.15, ge=0.01, le=1.0)
    X: float = Field(default=2.0, ge=0.01, le=20, description="Biomass concentration (g/L)")
    DO: float = Field(default=4.0, ge=0, le=20)
    temperature: float = Field(default=20, ge=0, le=50)
    pH: float = Field(default=7.5, ge=0, le=14)


class OxygenRequest(BaseModel):
    NH4_removed: float = Field(default=50, ge=0, le=10000, description="NH4-N removed per day (kg)")


class SensitivityReq(BaseModel):
    base_params: dict
    param_name: str
    values: list[float]
    S_init: float = 50
    X_init: float = 0.1
    time_days: float = 10


class ReactorComparisonRequest(BaseModel):
    S_init: float = Field(default=50, ge=1, le=1000)
    X_init: float = Field(default=0.1, ge=0.01, le=10)
    S_in: float = Field(default=50, ge=1, le=1000, description="CSTR influent NH4-N")
    Q: float = Field(default=1000, ge=1, le=1e6, description="Flow rate (m3/day)")
    V: float = Field(default=1000, ge=1, le=1e6, description="Reactor volume (m3)")
    mu_max: float = Field(default=0.8, ge=0.01, le=5.0)
    Ks: float = Field(default=2.0, ge=0.1, le=100)
    Y: float = Field(default=0.15, ge=0.01, le=1.0)
    temperature: float = Field(default=20, ge=0, le=50)
    pH: float = Field(default=7.5, ge=0, le=14)
    DO: float = Field(default=4.0, ge=0, le=20)
    time_days: float = Field(default=20, ge=1, le=100)


# ---- Routes ----

@app.get("/", response_class=HTMLResponse)
async def root():
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return HTMLResponse(content=static_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Nitrification Dynamics Simulator</h1><p>Frontend not found.</p>")


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "Nitrification Dynamics Simulator v3"}


@app.post("/api/predict")
async def predict(req: SimulationRequest):
    try:
        from app.models import solve_nitrification, normalize_params, compute_effective_mu_max
        import numpy as np
        params = normalize_params({
            "mu_max": req.mu_max, "Ks": req.Ks, "Y": req.Y,
            "inhibitor": req.inhibitor, "KI": req.KI,
            "inhibition_type": req.inhibition_type,
            "temperature": req.temperature, "pH": req.pH, "DO": req.DO,
        })
        y0, t_span, t_eval = [req.S_init, req.X_init], (0, req.time_days), np.linspace(0, req.time_days, 200)
        sol = solve_nitrification(y0, params, t_span, t_eval)
        mu_eff = compute_effective_mu_max(params)
        final_S = max(0, sol.y[0][-1])
        return {
            "success": True, "message": "Simulation completed",
            "data": {
                "time": sol.t.tolist(),
                "S": [max(0, v) for v in sol.y[0].tolist()],
                "X": [max(0, v) for v in sol.y[1].tolist()],
                "final_S": final_S, "final_X": max(0, sol.y[1][-1]),
                "removal_pct": round((1 - final_S / req.S_init) * 100, 2) if req.S_init > 0 else 0,
                "effective_mu_max": round(mu_eff, 4),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "message": "Simulation failed", "error": str(e)}


@app.post("/api/predict/aob-nob")
async def predict_aob_nob(req: AOBNoBRequest):
    try:
        from app.models import solve_nitrification, normalize_params, free_ammonia, free_nitrous_acid, fa_fna_warning
        import numpy as np
        params = normalize_params({
            "model_type": "aob_nob",
            "mu_max_AOB": req.mu_max_AOB, "K_NH4": req.K_NH4, "Y_AOB": req.Y_AOB,
            "mu_max_NOB": req.mu_max_NOB, "K_NO2": req.K_NO2, "Y_NOB": req.Y_NOB,
            "temperature": req.temperature, "pH": req.pH, "DO": req.DO,
        })
        y0 = [req.NH4_init, req.NO2_init, req.NO3_init, req.X_AOB_init, req.X_NOB_init]
        t_span, t_eval = (0, req.time_days), np.linspace(0, req.time_days, 200)
        sol = solve_nitrification(y0, params, t_span, t_eval)
        NH4, NO2, NO3, X_AOB, X_NOB = sol.y
        FA_vals = [free_ammonia(max(0, v), req.pH, req.temperature) for v in NH4]
        FNA_vals = [free_nitrous_acid(max(0, v), req.pH, req.temperature) for v in NO2]
        FA_final = free_ammonia(max(0, NH4[-1]), req.pH, req.temperature)
        FNA_final = free_nitrous_acid(max(0, NO2[-1]), req.pH, req.temperature)
        warnings = fa_fna_warning(FA_final, FNA_final)
        return {
            "success": True,
            "data": {
                "time": sol.t.tolist(),
                "NH4": [max(0, v) for v in NH4.tolist()],
                "NO2": [max(0, v) for v in NO2.tolist()],
                "NO3": [max(0, v) for v in NO3.tolist()],
                "X_AOB": [max(0, v) for v in X_AOB.tolist()],
                "X_NOB": [max(0, v) for v in X_NOB.tolist()],
                "FA": FA_vals, "FNA": FNA_vals,
                "FA_final": round(FA_final, 4),
                "FNA_final": round(FNA_final, 6),
                "FA_FNA_warnings": warnings,
                "NH4_removal": round((1 - max(0, NH4[-1]) / req.NH4_init) * 100, 2) if req.NH4_init > 0 else 0,
                "NO2_peak": round(max(NO2), 2),
                "NO3_final": round(max(0, NO3[-1]), 2),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/sensitivity")
async def sensitivity(req: SensitivityReq):
    try:
        from app.models import normalize_params, sensitivity_analysis
        base = normalize_params(req.base_params.copy())
        results = sensitivity_analysis(base, req.param_name, req.values, [req.S_init, req.X_init], (0, req.time_days))
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/export")
async def export_csv(S_init=50, X_init=0.1, time_days=10, mu_max=0.8, Ks=2.0, Y=0.15,
                     inhibition_type="none", inhibitor=0, KI=100, temperature=20, pH=7.5, DO=4.0):
    try:
        from app.models import solve_nitrification, normalize_params
        import numpy as np
        params = normalize_params({"mu_max": mu_max, "Ks": Ks, "Y": Y, "inhibitor": inhibitor, "KI": KI,
                                   "inhibition_type": inhibition_type, "temperature": temperature, "pH": pH, "DO": DO})
        sol = solve_nitrification([S_init, X_init], params, (0, time_days), np.linspace(0, time_days, 200))
        lines = ["time_days,NH4_mgL,Biomass_gL"]
        for t, s, x in zip(sol.t, sol.y[0], sol.y[1]):
            lines.append(f"{t:.4f},{max(0,s):.4f},{max(0,x):.4f}")
        return PlainTextResponse("\n".join(lines), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=nitrification_simulation.csv"})
    except Exception as e:
        return PlainTextResponse(f"Error: {e}", 400)


@app.post("/api/kinetic-analysis")
async def kinetic_analysis_endpoint(time: list[float] = Form(...), conc: list[float] = Form(...)):
    try:
        from app.models import kinetic_analysis
        result = kinetic_analysis(time, conc)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/parameter-estimation")
async def parameter_estimation(file: UploadFile = File(...), model_type: str = Form("monod")):
    try:
        from app.models import estimate_nitrification_params
        if model_type not in ("monod", "haldane"):
            return {"success": False, "error": "model_type must be 'monod' or 'haldane'"}
        content = await file.read()
        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        time_col = conc_col = None
        for col in reader.fieldnames or []:
            cl = col.lower().strip()
            if cl in ("time", "t", "time (hr)", "time (day)", "hour", "hr"):
                time_col = col
            if cl in ("nh4-n", "nh4", "s", "substrate", "conc", "concentration", "nh4_n", "nh4n"):
                conc_col = col
        if not time_col or not conc_col:
            return {"success": False, "error": "CSV must have time and NH4-N columns"}
        t_data, c_data = [], []
        for row in reader:
            try:
                t_data.append(float(row[time_col]))
                c_data.append(float(row[conc_col]))
            except (ValueError, KeyError):
                continue
        if len(t_data) < 3:
            return {"success": False, "error": "Need at least 3 data points"}
        result = estimate_nitrification_params(t_data, c_data, model_type)
        if not result.get("success"):
            return result
        result["time"] = t_data
        result["experimental"] = c_data
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/design-reactor")
async def design_reactor_endpoint(req: DesignRequest):
    try:
        from app.models import design_reactor
        res = design_reactor(req.S_in, req.Q, req.target_S_eff, req.mu_max, req.Ks, req.Y, req.X, req.DO, req.temperature, req.pH)
        return {"success": True, **res}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/reactor-comparison")
async def reactor_comparison(req: ReactorComparisonRequest):
    try:
        from app.models import solve_nitrification, normalize_params, cstr_steady_state, solve_cstr
        import numpy as np
        t_span, t_eval = (0, req.time_days), np.linspace(0, req.time_days, 300)

        # Batch reactor
        params_batch = normalize_params({
            "mu_max": req.mu_max, "Ks": req.Ks, "Y": req.Y,
            "temperature": req.temperature, "pH": req.pH, "DO": req.DO,
        })
        sol_batch = solve_nitrification([req.S_init, req.X_init], params_batch, t_span, t_eval)
        final_S_batch = max(0, sol_batch.y[0][-1])

        # CSTR dynamic
        params_cstr = normalize_params({
            "mu_max": req.mu_max, "Ks": req.Ks, "Y": req.Y,
            "temperature": req.temperature, "pH": req.pH, "DO": req.DO,
        })
        params_cstr.update({"S_in": req.S_in, "Q": req.Q, "V": req.V})
        HRT = req.V / req.Q
        sol_cstr = solve_cstr([req.S_in, req.X_init], params_cstr, t_span, t_eval)
        final_S_cstr = max(0, sol_cstr.y[0][-1])

        # CSTR steady-state
        ss = cstr_steady_state(params_cstr, HRT)

        return {
            "success": True,
            "data": {
                "time": sol_batch.t.tolist(),
                "batch_S": [max(0, v) for v in sol_batch.y[0].tolist()],
                "batch_X": [max(0, v) for v in sol_batch.y[1].tolist()],
                "cstr_S": [max(0, v) for v in sol_cstr.y[0].tolist()],
                "cstr_X": [max(0, v) for v in sol_cstr.y[1].tolist()],
                "HRT_days": round(HRT, 3),
                "HRT_hours": round(HRT * 24, 1),
                "batch_final_S": round(final_S_batch, 2),
                "batch_removal": round((1 - final_S_batch / req.S_init) * 100, 1) if req.S_init > 0 else 0,
                "cstr_final_S": final_S_cstr,
                "cstr_ss_S": ss["S_eff"],
                "cstr_ss_X": ss["X"],
                "cstr_removal": ss["removal_pct"],
                "cstr_mu_effective": ss["mu_effective"],
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/oxygen-demand")
async def oxygen_demand_endpoint(req: OxygenRequest):
    try:
        from app.models import oxygen_demand
        res = oxygen_demand(req.NH4_removed)
        return {"success": True, **res}
    except Exception as e:
        return {"success": False, "error": str(e)}
