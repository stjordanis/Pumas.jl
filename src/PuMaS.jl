module PuMaS

using DiffEqBase, DiffEqMonteCarlo, Reexport,
      StaticArrays, DiffEqJump, Distributed, LabelledArrays, GLM,
      TreeViews, CSV, DelayDiffEq

@reexport using OrdinaryDiffEq, Unitful
@reexport using Distributions, PDMats, DataFrames

const Numeric = Union{AbstractVector{<:Number}, Number}

include("data_parsing/data_types.jl")
include("data_parsing/data_read.jl")

include("lang/params.jl")
include("lang/model.jl")
include("lang/randoms.jl")
include("lang/parse.jl")

include("analytical_solutions/analytical_problem.jl")
include("analytical_solutions/analytical_solution_type.jl")
include("analytical_solutions/standard_models.jl")

include("simulate_methods/utils.jl")
include("simulate_methods/diffeqs.jl")
include("simulate_methods/analytical.jl")

include("nca/auc.jl")

example_nmtran_data(filename) = joinpath(joinpath(@__DIR__, ".."),"examples/"*filename*".csv")

export Subject, Population, process_nmtran, DosageRegimen
export PKPDModel, init_param, init_random, rand_random,
       simobs, likelihood, pre, simpost,
       AUC, AUMC
export example_nmtran_data
end # module
