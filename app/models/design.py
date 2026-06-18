import numpy as np


def design_reactor(S_in, Q, target_S_eff, mu_max, Ks, Y, X, DO, T=20, pH=7.5):
    mu = mu_max * (S_in / (Ks + S_in))
    T_factor = 1.07 ** (T - 20)
    pH_factor = 1.0 if 6.5 <= pH <= 8.5 else np.exp(-0.5 * ((pH - 6.5 if pH < 6.5 else pH - 8.5) / 1.5) ** 2)
    DO_factor = DO / (0.5 + DO)
    mu *= T_factor * pH_factor * DO_factor

    HRT = (S_in - target_S_eff) * Y / (mu * X) if X > 0 and mu > 0 else 0
    V = HRT * Q

    return {
        "HRT_days": round(HRT, 3),
        "HRT_hours": round(HRT * 24, 1),
        "volume_m3": round(V, 1),
        "organic_loading": round(S_in * Q / V, 3) if V > 0 else 0,
        "mu_effective": round(mu, 4),
        "removal_pct": round((1 - target_S_eff / S_in) * 100, 1) if S_in > 0 else 0,
    }


def oxygen_demand(NH4_removed, SRT_effect=None):
    O2_per_NH4 = 4.57
    total_O2 = NH4_removed * O2_per_NH4

    O2_in_air = 0.23
    transfer_efficiency = 0.20
    air_required = total_O2 / (O2_in_air * transfer_efficiency)

    energy_per_m3_air = 0.3
    energy_kWh = air_required * energy_per_m3_air / 1000

    return {
        "oxygen_demand_kg": round(total_O2 / 1000, 2),
        "oxygen_demand_g": round(total_O2, 1),
        "air_required_m3": round(air_required, 1),
        "energy_estimate_kWh": round(energy_kWh, 2),
        "note": "Based on 4.57 mgO2/mgNH4-N stoichiometry",
    }
