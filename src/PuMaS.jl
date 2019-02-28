module PuMaS

using DiffEqBase, DiffEqMonteCarlo, Reexport,
      StaticArrays, DiffEqJump, Distributed, LabelledArrays, GLM,
      TreeViews, CSV, DelayDiffEq, ForwardDiff, DiffResults, Optim,
      Missings, RecipesBase, StructArrays

@reexport using OrdinaryDiffEq, Unitful
@reexport using Distributions, PDMats, DataFrames

const Numeric = Union{AbstractVector{<:Number}, Number}

include("data_parsing/data_types.jl")
include("data_parsing/data_read.jl")

include("dsl/model_macro.jl")

include("models/params.jl")
include("models/simulated_observations.jl")
include("models/model_api.jl")
include("models/transforms.jl")
include("models/likelihoods.jl")
include("models/bayes.jl")

include("analytical_solutions/analytical_problem.jl")
include("analytical_solutions/analytical_solution_type.jl")
include("analytical_solutions/standard_models.jl")

include("simulate_methods/utils.jl")
include("simulate_methods/diffeqs.jl")
include("simulate_methods/analytical.jl")

include("nca/NCA.jl")

@reexport using .NCA

example_nmtran_data(filename) = joinpath(joinpath(@__DIR__, ".."),"examples/"*filename*".csv")

export Subject, Population, process_nmtran, DosageRegimen
export PKPDModel, init_param, init_random, rand_random,
       simobs, pre, simpost
export conditional_nll, ll_derivatives, FIM, npde, wres
export example_nmtran_data
export @model
end # module
