"""
This is a very simple approach to counting the number of lines
per approach (explicit, implicit, DiffEq).

Core structs and functions in the source code have been annotated
with entries like [core], [jacobian], etc. The functions here
count the number of lines, skipping empty lines, comments, etc.

Some functions may have multiple annotations, e.g. some functions
are used by both explicit and implicit components:

# [explicit]
# [implicit]
function f()
    return
end

In this case, the line count function should be supplied
one of both annotations. If both are supplied, it will
be counted to the first one encountered.
"""

using CSV
using DataFrames
using Glob


const TRIPLE_QUOTE = "\"\"\""

function valid(line)
    stripped = strip(line)
    if isempty(stripped) || startswith(stripped, "#")
        return false
    else
        return true
    end
end

function advance_docstring(lines, i, nline)
    while i <= nline
        line = lines[i]
        if startswith(strip(line), TRIPLE_QUOTE)
            return i
        end
        i += 1
    end
    return i
end

function advance(lines, i, nline)
    count = 0
    while i <= nline
        # No whitespace indentation, marks global scope.
        line = lines[i]

        if startswith(strip(line), TRIPLE_QUOTE)
            i = advance_docstring(lines, i + 1, nline)
        else
            if valid(line)
                count += 1

                if length(line) >= 3 && line[1:3] == "end"
                    return count, i
                end
            end
        end

        i += 1
    end
    return count, i
end

function line_count(paths, annotations)
    counts = Dict((key, 0) for key in annotations)

    for path in paths
        lines = readlines(open(path, "r"))
        nline = length(lines)
        i = 1
        while i <= nline
            line = lines[i]
            for key in keys(counts)
                if startswith(line, "# [$(key)]")
                    count, i = advance(lines, i, nline)
                    counts[key] = counts[key] + count
                end
            end
            i += 1
        end
    end

    return counts
end


function case_count(case_paths)
    explicit_annotations = ["forcing", "core", "explicit", "output"]
    implicit_annotations =
        ["forcing", "core", "jacobian", "implicit", "nonlinear_solve", "output"]
    diffeq_annotations = ["forcing", "core", "diffeq", "output"]

    diffeq_linecount = line_count(
        [
            "src/forcing.jl";
            "src/model/model.jl";
            "src/model/model_diffeq.jl";
            glob("src/solver/*.jl");
            case_paths
        ],
        diffeq_annotations,
    )
    # The Jacobian sparsity handling for DiffEq is 9 lines.
    diffeq_linecount["diffeq"] = diffeq_linecount["diffeq"] - 9
    diffeq_linecount["jacobian"] = 9

    return Dict(
        "Explicit" => line_count(
            [
                "src/forcing.jl";
                "src/model/model.jl";
                "src/model/model_explicit.jl";
                glob("src/solver/*.jl");
                case_paths
            ],
            explicit_annotations,
        ),
        "Implicit" => line_count(
            [
                "src/forcing.jl";
                "src/model/model.jl";
                "src/model/model_implicit.jl";
                glob("src/solver/*.jl");
                case_paths
            ],
            implicit_annotations,
        ),
        "DiffEq" => diffeq_linecount,
    )
end

function create_counts_dataframe(counts)
    # Model order
    model_order = ["Reservoirs", "FUSE-070", "Richards"]

    rows = []

    # Shared across models
    push!(rows, ("Shared across models:", "", "", ""))
    push!(
        rows,
        (
            "Nonlinear solver",
            "",
            string(counts["Reservoirs"]["Implicit"]["nonlinear_solve"]),
            "",
        ),
    )
    push!(
        rows,
        (
            "Forcing + Output",
            string(
                counts["Reservoirs"]["Explicit"]["forcing"] +
                counts["Reservoirs"]["Explicit"]["output"],
            ),
            string(
                counts["Reservoirs"]["Implicit"]["forcing"] +
                counts["Reservoirs"]["Implicit"]["output"],
            ),
            string(
                counts["Reservoirs"]["DiffEq"]["forcing"] +
                counts["Reservoirs"]["DiffEq"]["output"],
            ),
        ),
    )

    # Each model
    for model in model_order
        explicit = counts[model]["Explicit"]
        implicit = counts[model]["Implicit"]
        diffeq = counts[model]["DiffEq"]

        push!(rows, ("$model", "", "", ""))
        push!(
            rows,
            (
                "Core",
                string(explicit["core"]),
                string(implicit["core"]),
                string(diffeq["core"]),
            ),
        )
        push!(
            rows,
            ("Jacobian", "", string(implicit["jacobian"]), string(diffeq["jacobian"])),
        )
        push!(
            rows,
            (
                "Time integration",
                string(explicit["explicit"]),
                string(implicit["implicit"]),
                string(diffeq["diffeq"]),
            ),
        )

        push!(
            rows,
            (
                "Total",
                string(sum(values(explicit))),
                string(sum(values(implicit))),
                string(sum(values(diffeq))),
            ),
        )
    end

    df = DataFrame(
        Component = [r[1] for r in rows],
        Explicit = [r[2] for r in rows],
        Implicit = [r[3] for r in rows],
        DiffEq = [r[4] for r in rows],
    )

    return df
end

counts = Dict(
    "Richards" => case_count(
        [glob("src/richards/*.jl"); "src/richards/constitutive/mualemvangenuchten.jl"],
    ),
    "FUSE-070" => case_count(glob("src/fuse/*.jl")),
    "Reservoirs" => case_count(glob("src/reservoirs/*.jl")),
)

df = create_counts_dataframe(counts)
@show df
CSV.write("cases/output/linecounts.csv", df)
