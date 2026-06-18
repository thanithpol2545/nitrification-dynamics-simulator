import numpy as np
from scipy.integrate import solve_ivp

from .inhibition import apply_inhibition


def temperature_correction(T, T_ref=20, theta=1.07):
    return theta ** (T - T_ref)


def ph_factor(pH, pH_opt_low=6.5, pH_opt_high=8.5):
    if pH <= 0:
        return 0.0
    if pH_opt_low <= pH <= pH_opt_high:
        return 1.0
    if pH < pH_opt_low:
        return np.exp(-0.5 * ((pH - pH_opt_low) / 1.5) ** 2)
    return np.exp(-0.5 * ((pH - pH_opt_high) / 1.5) ** 2)


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
    mu_max = params["mu_max"]
    Ks = params["Ks"]
    Y = params["Y"]
    I = params.get("inhibitor", 0)
    KI = params.get("KI", 100)
    inhibition_type = params.get("inhibition_type", "none")

    mu = apply_inhibition(S, mu_max, Ks, Y, I, KI, inhibition_type)

    T = params.get("temperature", 20)
    T_ref = params.get("T_ref", 20)
    theta = params.get("theta", 1.07)
    mu *= temperature_correction(T, T_ref, theta)

    pH = params.get("pH", 7.5)
    mu *= ph_factor(pH)

    dS_dt = -(mu * X / Y)
    dX_dt = mu * X

    return [dS_dt, dX_dt]


def solve_nitrification(init_cond, params, t_span, t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)

    inhibition_type = params.get("inhibition_type", "none")
    model = (
        kinetic_model_with_inhibition
        if inhibition_type != "none"
        else kinetic_model
    )

    sol = solve_ivp(
        model,
        t_span,
        init_cond,
        args=(params,),
        t_eval=t_eval,
        method="RK45",
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol


def compute_effective_mu_max(params):
    mu_max = params["mu_max"]
    T = params.get("temperature", 20)
    T_ref = params.get("T_ref", 20)
    theta = params.get("theta", 1.07)
    pH = params.get("pH", 7.5)
    return mu_max * temperature_correction(T, T_ref, theta) * ph_factor(pH)


def sensitivity_analysis(base_params, param_name, values, init_cond, t_span, t_eval=None):
    results = []
    for val in values:
        test_params = base_params.copy()
        test_params[param_name] = val
        sol = solve_nitrification(init_cond, test_params, t_span, t_eval)
        max_mu_eff = compute_effective_mu_max(test_params)
        results.append({
            "param_value": val,
            "final_S": max(0, sol.y[0][-1]),
            "final_X": max(0, sol.y[1][-1]),
            "removal_pct": round((1 - max(0, sol.y[0][-1]) / init_cond[0]) * 100, 2) if init_cond[0] > 0 else 0,
            "effective_mu_max": round(max_mu_eff, 4),
        })
    return results


def normalize_params(params):
    params["mu_max"] = max(0.01, min(5.0, params["mu_max"]))
    params["Ks"] = max(0.1, min(100, params["Ks"]))
    params["Y"] = max(0.01, min(1.0, params["Y"]))
    params["KI"] = max(0.1, min(1000, params.get("KI", 100)))
    params["inhibitor"] = max(0, params.get("inhibitor", 0))
    params["temperature"] = max(0, min(50, params.get("temperature", 20)))
    params["pH"] = max(0, min(14, params.get("pH", 7.5)))
    return params
