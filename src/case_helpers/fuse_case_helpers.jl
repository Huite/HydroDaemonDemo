START_DATE = Date(1997, 8, 1)
END_DATE = Date(2001, 9, 30)

function read_rainfall(path)
    df = CSV.read(
        path,
        DataFrame,
        header = 2,
        missingstring = "GAP",
        dateformat = "d-u-yyyy",
    )
    # The time is at 24:00:00 so we offset it by one day.
    mask24 = df.Time .== "24:00:00"
    df.Date[mask24] .+= Day(1)
    df.Time[mask24] .= "00:00:00"
    return select(df, :Date, Symbol("Rain mm"))
end

function read_evaporation(path)
    df = CSV.read(path, DataFrame, dateformat = dateformat"yyyy-mm-ddTHH:MM:SSZ")
    df."Date" = Date.(df."Observation time UTC")
    df.Evaporation = df."Evaporation [mm]"
    return select(df, :Date, Symbol("Evaporation"))
end

function create_mahurangi_forcingdf(rainfall_globpath, pet_path)
    daily_files = glob(rainfall_globpath)
    # Skip station 644625 since it has only 581 rows.
    # This results in 13 stations, though coverage at the start and end is spotty.
    # Some locations started later, or stopped earlier.
    dfs = filter(df -> nrow(df) > 600, [read_rainfall(path) for path in daily_files])
    forcingdf = DataFrame(Date = START_DATE:END_DATE)
    forcingdf = reduce(
        (acc, df) -> leftjoin(acc, df, on = :Date, makeunique = true),
        dfs;
        init = forcingdf,
    )

    # Row-wise mean, ignoring missing.
    forcingdf.:Precipitation =
        [mean(skipmissing(row)) for row in eachrow(forcingdf[:, Not(:Date)])]

    # Add Priestly-Taylor PET, set missing to 0.
    petdf = read_evaporation(pet_path)
    forcingdf = leftjoin(forcingdf, petdf, on = :Date, makeunique = true)
    replace!(forcingdf.Evaporation, missing => 0)

    sort!(forcingdf, :Date)
    return forcingdf
end

function read_mahurangi_streamflow(path)
    seconds_in_day = 86_400.0
    mahurangi_area = 46.65e6  # 46.65 km^2
    df = CSV.read(path, DataFrame, header = 2, missingstring = "GAP", dateformat = "d-u-y")
    # The time is at 24:00:00 so we offset it by one day.
    mask24 = df.Time .== "24:00:00"
    df.Date[mask24] .+= Day(1)
    df.discharge = @. df."Flow l/s" * seconds_in_day / mahurangi_area
    # Keep missing as is.
    discharge_df = DataFrame(Date = START_DATE:END_DATE)
    discharge_df = leftjoin(discharge_df, df, on = :Date)
    return select(discharge_df, :Date, :discharge)
end

function create_gamma_kernel(n, μτ; a = 3.0)
    gamma_dist = Gamma(a, μτ/a)
    # Generate weights for each time step
    weights = zeros(Float64, n)
    for i = 1:n
        weights[i] = cdf(gamma_dist, Float64(i)) - cdf(gamma_dist, Float64(i-1))
    end
    # Normalize to ensure sum = 1
    return weights ./ sum(weights)
end

function route_flows(flows, μτ)
    # Convolve the flow with a Gamma kernel.
    kernel = create_gamma_kernel(length(flows), μτ)
    n_flows = length(flows)
    routed = zeros(n_flows)
    for i = 1:n_flows
        for j = 1:min(i, n_flows)
            routed[i] += flows[i-j+1] * kernel[j]
        end
    end
    return routed
end
