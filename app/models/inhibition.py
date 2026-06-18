def inhibition_factor(I, KI, inhibition_type="none"):
    if inhibition_type == "none":
        return 1.0
    if KI <= 0:
        return 1.0
    factor = 1 + (I / KI)
    if inhibition_type == "competitive":
        return factor
    elif inhibition_type == "uncompetitive":
        return factor
    elif inhibition_type == "non-competitive":
        return factor
    return 1.0


def apply_inhibition(S, mu_max, Ks, Y, I, KI, inhibition_type):
    inh_factor = inhibition_factor(I, KI, inhibition_type)

    if inhibition_type == "competitive":
        mu = mu_max * (S / (Ks * inh_factor + S))
    elif inhibition_type == "uncompetitive":
        mu = mu_max * (S / (inh_factor * (Ks + S)))
    elif inhibition_type == "non-competitive":
        mu = mu_max * (S / (Ks + S * inh_factor))
    else:
        mu = mu_max * (S / (Ks + S))

    return mu
