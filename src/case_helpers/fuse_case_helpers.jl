const Bounds = NamedTuple{(:lower, :upper),Tuple{Float64,Float64}}

"""
This implements only the bounds required by FUSE070 and FUSE550 models.
"""
struct FuseParameterBounds
    S1max::Bounds
    S2max::Bounds
    ϕtens::Bounds
    r1::Bounds
    ku::Bounds
    c::Bounds
    ki::Bounds
    ks::Bounds
    n::Bounds
    v::Bounds
    Acmax::Bounds
    b::Bounds
    μt::Bounds
end

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
    start_date = Date(1997, 8, 1)
    end_date = Date(2001, 9, 30)

    daily_files = glob(rainfall_globpath)
    # Skip station 644625 since it has only 581 rows.
    # This results in 13 stations, though coverage at the start and end is spotty.
    # Some locations started later, or stopped earlier.
    dfs = filter(df -> nrow(df) > 600, [read_rainfall(path) for path in daily_files])
    forcingdf = DataFrame(Date = start_date:end_date)
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
