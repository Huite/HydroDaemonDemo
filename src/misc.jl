## 
using CSV
using DataFrames
using Dates

function read_weather_data(filename)
    # Read the CSV file, skipping the header until we find the column names
    df = CSV.read(
        filename,
        DataFrame,
        header = 52,
        skipto = 54,
        comment = "# ",  # Treat rows starting with # as comments
        silencewarnings = true,
    )  # Silence warnings about type inference
    # Strip whitespace from column names
    rename!(df, [Symbol(strip(s)) for s in names(df)])
    return df
end

# Usage
filename = "data/etmgeg_260.txt"
weather = read_weather_data(filename)

df_selected = select(weather, [:YYYYMMDD, :RH, :EV24])
df_transformed = transform(
    df_selected,
    :YYYYMMDD => ByRow(x -> Date(string(x), dateformat"yyyymmdd")) => :Date,
    :RH =>
        ByRow(x -> coalesce(max(0, something(tryparse(Float, x), -1)) / 10.0, 0)) => :P,
    :EV24 =>
        ByRow(x -> coalesce(max(0, something(tryparse(Float, x), -1)) / 10.0, 0)) =>
            :ET,
)
df = select(
    filter(row -> row.Date >= Dates.Date("1960-01-01"), df_transformed),
    [:Date, :P, :ET],
)
CSV.write("forcing.csv", df)
