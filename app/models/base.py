import numpy as np
from scipy.integrate import solve_ivp

from .inhibition import apply_inhibition


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


def normalize_params(params):
    params["mu_max"] = max(0.01, min(5.0, params["mu_max"]))
    params["Ks"] = max(0.1, min(100, params["Ks"]))
    params["Y"] = max(0.01, min(1.0, params["Y"]))
    params["KI"] = max(0.1, min(1000, params.get("KI", 100)))
    params["inhibitor"] = max(0, params.get("inhibitor", 0))
    return params
