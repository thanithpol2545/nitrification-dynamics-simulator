import math


def zero_order_model(t, C0, k):
    return C0 - k * t


def first_order_model(t, C0, k):
    return C0 * math.exp(-k * t)


def second_order_model(t, C0, k):
    return C0 / (1 + C0 * k * t)


def kinetic_analysis(time_data, conc_data):
    import numpy as np
    from scipy.optimize import curve_fit
    t = np.array(time_data, dtype=float)
    C = np.array(conc_data, dtype=float)

    results = []
    models = [
        ("Zero-order", zero_order_model, ["C0", "k"]),
        ("First-order", first_order_model, ["C0", "k"]),
        ("Second-order", second_order_model, ["C0", "k"]),
    ]

    for name, func, _ in models:
        try:
            popt, _ = curve_fit(func, t, C, p0=[C[0], 0.1], maxfev=5000)
            C_pred = func(t, *popt)
            ss_res = np.sum((C - C_pred) ** 2)
            ss_tot = np.sum((C - np.mean(C)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((C - C_pred) ** 2))
            results.append({
                "model": name,
                "C0": round(popt[0], 4),
                "k": round(popt[1], 6),
                "R2": round(r2, 4),
                "RMSE": round(rmse, 4),
                "predicted": [round(float(v), 4) for v in C_pred],
            })
        except Exception:
            results.append({"model": name, "C0": 0, "k": 0, "R2": 0, "RMSE": 999, "predicted": []})

    best = max(results, key=lambda r: r["R2"])
    return {"results": results, "best_model": best["model"], "best_R2": best["R2"]}


def estimate_nitrification_params(time_data, conc_data, model_type="monod"):
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.integrate import solve_ivp
    t = np.array(time_data, dtype=float)
    C = np.array(conc_data, dtype=float)

    def monod_ode(t, S0, mu_max, Ks, Y, X0):
        def ode(_, y):
            S, X = y
            mu = mu_max * (S / (Ks + S))
            return [-(mu * X / Y), mu * X]
        sol = solve_ivp(ode, (0, t.max()), [S0, X0], t_eval=t, method="RK45")
        return sol.y[0]

    def haldane_ode(t, S0, mu_max, Ks, Ki, Y, X0):
        def ode(_, y):
            S, X = y
            mu = mu_max * S / (Ks + S + S**2 / Ki) if Ki > 0 else 0
            return [-(mu * X / Y), mu * X]
        sol = solve_ivp(ode, (0, t.max()), [S0, X0], t_eval=t, method="RK45")
        return sol.y[0]

    try:
        if model_type == "haldane":
            popt, _ = curve_fit(
                lambda t, mu_max, Ks, Ki, Y: haldane_ode(t, C[0], mu_max, Ks, Ki, Y, 0.1),
                t, C, p0=[1.0, 5.0, 50, 0.15],
                bounds=([0.01, 0.1, 1, 0.01], [5.0, 100, 500, 0.5]),
                maxfev=10000,
            )
            C_pred = haldane_ode(t, C[0], *popt, 0.1)
            result = {
                "mu_max": round(popt[0], 4),
                "Ks": round(popt[1], 4),
                "Ki": round(popt[2], 4),
                "Y": round(popt[3], 4),
                "model_type": "haldane",
            }
        else:
            popt, _ = curve_fit(
                lambda t, mu_max, Ks, Y: monod_ode(t, C[0], mu_max, Ks, Y, 0.1),
                t, C, p0=[0.8, 2.0, 0.15], bounds=([0.01, 0.1, 0.01], [5.0, 50, 0.5]),
                maxfev=10000,
            )
            C_pred = monod_ode(t, C[0], *popt, 0.1)
            result = {
                "mu_max": round(popt[0], 4),
                "Ks": round(popt[1], 4),
                "Y": round(popt[2], 4),
                "model_type": "monod",
            }

        ss_res = np.sum((C - C_pred) ** 2)
        ss_tot = np.sum((C - np.mean(C)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((C - C_pred) ** 2))
        result.update({
            "success": True,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "predicted": [round(float(v), 4) for v in C_pred],
        })
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
