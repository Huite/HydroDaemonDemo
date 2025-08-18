function waterbalance!(dS, S, fuse::Fuse550Parameters)
    (; ϕtens, S1max, S2max, r1, r2, ku, ki, ks, c, n, Acmax, ω, currentforcing) = fuse
    p = currentforcing[1]
    PET = currentforcing[2]

    S1 = S[1]
    S2 = S[2]

    # Compute saturation variables
    # Ensure saturation is [0.0 - 1.0], since exponentiation of a negative
    # number results in a domain error (requires complex numbers).
    S1_sat = clamp_smooth(S1, 0.0, S1max, ω) / S1max
    S1_sat_free = S1 / (1.0 - ϕtens) * S1max
    # saturation of tension storage may be > 1.0 ("super-saturation")
    S1_sat_tens = S1 / (ϕtens * S1max)
    S2_sat_tens = S2 / (ϕtens * S2max)

    # Runoff
    qsx = p * S1_sat_tens * Acmax

    # Evaporation
    e1 = PET * r1 * dmin_smooth(S1_sat_tens, 1.0, FUSE_RHO)
    e2 = PET * r2 * dmin_smooth(S2_sat_tens, 1.0, FUSE_RHO)

    # Drainage
    q12 = ku * S1_sat^c

    # Interflow
    qif = ki * S1_sat_free

    # Baseflow
    qb = ks * (S2 / S2max) ^ n

    # overflow
    qufof = (p - qsx) * sigmoid_activation(S1, S1max, ω)
    qsfof = q12 * sigmoid_activation(S2, S2max, ω)

    dS[1] = p - qsx - e1 - q12 - qif - qufof
    dS[2] = q12 - e2 - qb - qsfof
    return e1 + e2, qsx + qif + qufof + qb + qsfof
end
