import numpy as np
from scipy.integrate import solve_ivp

from .inhibition import apply_inhibition

# ---- Environmental factors ----

def temperature_factor(T, T_ref=20, theta=1.07):
    return theta ** (T - T_ref)


def ph_factor(pH):
    if pH <= 0:
        return 0.0
    if 6.5 <= pH <= 8.5:
        return 1.0
    if pH < 6.5:
        return np.exp(-0.5 * ((pH - 6.5) / 1.5) ** 2)
    return np.exp(-0.5 * ((pH - 8.5) / 1.5) ** 2)


def do_factor(DO, K_DO):
    return DO / (K_DO + DO) if DO > 0 else 0


# ---- FA / FNA calculation ----

def free_ammonia(NH4, pH, T):
    if NH4 <= 0:
        return 0.0
    pKa = 9.25
    return NH4 * 10 ** (pH - pKa) / (1 + 10 ** (pH - pKa))


def free_nitrous_acid(NO2, pH, T):
    if NO2 <= 0:
        return 0.0
    pKa = 3.35
    return NO2 / (1 + 10 ** (pH - pKa))


def fa_fna_warning(FA, FNA):
    warnings = {"fa": {"value": round(FA, 4), "level": "low", "message": ""},
                "fna": {"value": round(FNA, 6), "level": "low", "message": ""}}
    if FA > 150:
        warnings["fa"]["level"] = "high"
        warnings["fa"]["message"] = "Strong AOB inhibition expected"
    elif FA > 10:
        warnings["fa"]["level"] = "moderate"
        warnings["fa"]["message"] = "Potential AOB inhibition"
    if FNA > 0.22:
        warnings["fna"]["level"] = "high"
        warnings["fna"]["message"] = "Strong NOB inhibition expected"
    elif FNA > 0.026:
        warnings["fna"]["level"] = "moderate"
        warnings["fna"]["message"] = "Potential NOB inhibition"
    return warnings


# ---- Single-step model (original) ----

def kinetic_model(t, y, params):
    S, X = y
    mu_max = params["mu_max"]
    Ks = params["Ks"]
    Y = params["Y"]
    mu = mu_max * (S / (Ks + S))
    dS_dt = -(mu * X / Y)
    dX_dt = mu * X
    return [dS_dt, dX_dt]


def kinetic_model_with_inhibition(t, y, params):
    S, X = y
    mu = apply_inhibition(S, params["mu_max"], params["Ks"], params["Y"],
                          params.get("inhibitor", 0), params.get("KI", 100),
                          params.get("inhibition_type", "none"))
    mu *= temperature_factor(params.get("temperature", 20))
    mu *= ph_factor(params.get("pH", 7.5))
    mu *= do_factor(params.get("DO", 4.0), params.get("K_DO", 0.5))
    dS_dt = -(mu * X / params["Y"])
    dX_dt = mu * X
    return [dS_dt, dX_dt]


# ---- Two-step AOB/NOB model ----

def aob_nob_model(t, y, params):
    NH4, NO2, NO3, X_AOB, X_NOB = y
    T = params.get("temperature", 20)
    pH_val = params.get("pH", 7.5)
    DO_val = params.get("DO", 4.0)

    tf = temperature_factor(T)
    pf = ph_factor(pH_val)

    mu_AOB = params["mu_max_AOB"] * (NH4 / (params["K_NH4"] + NH4)) * tf * pf * do_factor(DO_val, params.get("K_DO_AOB", 0.3))
    mu_NOB = params["mu_max_NOB"] * (NO2 / (params["K_NO2"] + NO2)) * tf * pf * do_factor(DO_val, params.get("K_DO_NOB", 0.5))

    FA = free_ammonia(NH4, pH_val, T)
    FNA = free_nitrous_acid(NO2, pH_val, T)
    fa_inh = params.get("KI_FA_AOB", 150) / (params.get("KI_FA_AOB", 150) + FA) if FA > 0 else 1.0
    fna_inh = params.get("KI_FNA_NOB", 0.5) / (params.get("KI_FNA_NOB", 0.5) + FNA) if FNA > 0 else 1.0
    mu_AOB *= fa_inh
    mu_NOB *= fna_inh

    r_AOB = mu_AOB * X_AOB
    r_NOB = mu_NOB * X_NOB

    dNH4 = -r_AOB / params["Y_AOB"]
    dNO2 = r_AOB / params["Y_AOB"] - r_NOB / params["Y_NOB"]
    dNO3 = r_NOB / params["Y_NOB"]
    dX_AOB = r_AOB
    dX_NOB = r_NOB

    return [dNH4, dNO2, dNO3, dX_AOB, dX_NOB]


# ---- CSTR steady-state ----

def cstr_steady_state(params, HRT):
    S_in = params.get("S_in", 50)
    mu_max = params.get("mu_max", params.get("mu_max_AOB", 0.8))
    Ks = params.get("Ks", params.get("K_NH4", 2.0))
    Y = params.get("Y", params.get("Y_AOB", 0.15))
    T = params.get("temperature", 20)
    pH_val = params.get("pH", 7.5)
    DO_val = params.get("DO", 4.0)
    D = 1.0 / HRT if HRT > 0 else 0
    mu_max_eff = mu_max * temperature_factor(T) * ph_factor(pH_val) * do_factor(DO_val, 0.5)
    if D >= mu_max_eff:
        S_eff = S_in
        X = 0.0
    else:
        S_eff = Ks * D / (mu_max_eff - D)
        S_eff = max(0.0, min(S_in, S_eff))
        X = Y * (S_in - S_eff)
    return {
        "S_eff": round(S_eff, 2),
        "X": round(max(X, 0), 4),
        "removal_pct": round((1 - S_eff / S_in) * 100, 1) if S_in > 0 else 0,
        "mu_effective": round(mu_max_eff, 4),
        "D": round(D, 4),
    }


# ---- CSTR dynamic (transient) model ----

def cstr_model(t, y, params):
    S, X = y
    S_in = params.get("S_in", 50)
    Q = params.get("Q", 1000)
    V = params.get("V", 1000)
    D = Q / V
    mu_max = params.get("mu_max", params.get("mu_max_AOB", 0.8))
    Ks = params.get("Ks", params.get("K_NH4", 2.0))
    Y = params.get("Y", params.get("Y_AOB", 0.15))
    mu = mu_max * (S / (Ks + S))
    T = params.get("temperature", 20)
    pH_val = params.get("pH", 7.5)
    DO_val = params.get("DO", 4.0)
    mu *= temperature_factor(T) * ph_factor(pH_val) * do_factor(DO_val, 0.5)
    dS_dt = D * (S_in - S) - (mu * X / Y)
    dX_dt = -D * X + mu * X
    return [dS_dt, dX_dt]


def solve_cstr(init_cond, params, t_span, t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)
    sol = solve_ivp(cstr_model, t_span, init_cond, args=(params,), t_eval=t_eval, method="RK45", vectorized=False)
    if not sol.success:
        raise RuntimeError(f"CSTR solver failed: {sol.message}")
    return sol


# ---- Solve wrappers ----

def solve_nitrification(init_cond, params, t_span, t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)

    if params.get("model_type") == "aob_nob":
        defaults = {"mu_max_AOB": 0.8, "K_NH4": 2.0, "Y_AOB": 0.15,
                    "mu_max_NOB": 1.0, "K_NO2": 1.5, "Y_NOB": 0.08,
                    "K_DO_AOB": 0.3, "K_DO_NOB": 0.5,
                    "KI_FA_AOB": 150, "KI_FNA_NOB": 0.5}
        for k, v in defaults.items():
            params.setdefault(k, v)
        sol = solve_ivp(aob_nob_model, t_span, init_cond, args=(params,),
                        t_eval=t_eval, method="RK45", vectorized=False)
    else:
        inhibition_type = params.get("inhibition_type", "none")
        model = kinetic_model_with_inhibition if inhibition_type != "none" else kinetic_model
        defaults = {"K_DO": 0.5}
        for k, v in defaults.items():
            params.setdefault(k, v)
        sol = solve_ivp(model, t_span, init_cond, args=(params,),
                        t_eval=t_eval, method="RK45", vectorized=False)

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol


def compute_effective_mu_max(params):
    mu = params.get("mu_max", params.get("mu_max_AOB", 0.8))
    T = params.get("temperature", 20)
    pH_val = params.get("pH", 7.5)
    DO_val = params.get("DO", 4.0)
    mu *= temperature_factor(T) * ph_factor(pH_val) * do_factor(DO_val, params.get("K_DO", 0.5))
    return mu


def sensitivity_analysis(base_params, param_name, values, init_cond, t_span, t_eval=None):
    results = []
    for val in values:
        p = base_params.copy()
        p[param_name] = val
        sol = solve_nitrification(init_cond, p, t_span, t_eval)
        final_S = max(0, sol.y[0][-1])
        results.append({
            "param_value": val,
            "final_S": final_S,
            "final_X": max(0, sol.y[-1][-1]) if p.get("model_type") == "aob_nob" else max(0, sol.y[1][-1]),
            "removal_pct": round((1 - final_S / init_cond[0]) * 100, 2) if init_cond[0] > 0 else 0,
            "effective_mu_max": round(compute_effective_mu_max(p), 4),
        })
    return results


def normalize_params(params):
    for k in ["mu_max", "mu_max_AOB", "mu_max_NOB"]:
        if k in params:
            params[k] = max(0.01, min(5.0, params[k]))
    for k in ["Ks", "K_NH4", "K_NO2"]:
        if k in params:
            params[k] = max(0.1, min(100, params[k]))
    for k in ["Y", "Y_AOB", "Y_NOB"]:
        if k in params:
            params[k] = max(0.01, min(1.0, params[k]))
    if "KI" in params:
        params["KI"] = max(0.1, min(1000, params["KI"]))
    if "inhibitor" in params:
        params["inhibitor"] = max(0, params.get("inhibitor", 0))
    if "temperature" in params:
        params["temperature"] = max(0, min(50, params["temperature"]))
    if "pH" in params:
        params["pH"] = max(0, min(14, params["pH"]))
    if "DO" in params:
        params["DO"] = max(0, min(20, params["DO"]))
    return params
