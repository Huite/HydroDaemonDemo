function prepare_state(p::Richards1dParameters, initial, forcing)
    n = length(p.constitutive)
    return RichardsExplicitState(
        initial,  # ψ   
        zeros(n),  # θ
        zeros(n),  # C
        zeros(n),  # k
        zeros(n),  # k_inter
        zeros(n),  # kΔz⁻¹
        zeros(n),  # Δψ
        forcing,
    )
end
