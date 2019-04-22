module PuMaS

using DiffEqBase, DiffEqMonteCarlo, Reexport,
      StaticArrays, DiffEqJump, Distributed, LabelledArrays, GLM,
      TreeViews, CSV, DelayDiffEq, ForwardDiff, DiffResults, Optim,
      Missings, RecipesBase, StructArrays, RecursiveArrayTools

@reexport using OrdinaryDiffEq, Unitful
@reexport using Distributions, PDMats, DataFrames, StatsBase

const Numeric = Union{AbstractVector{<:Number}, Number}

function opt_minimizer end
include("nca/NCA.jl")

include("data_parsing/data_types.jl")
include("data_parsing/data_read.jl")

include("dsl/model_macro.jl")

include("models/params.jl")
include("models/simulated_observations.jl")
include("models/model_api.jl")
include("models/derived_utils.jl")

include("estimation/transforms.jl")
include("estimation/likelihoods.jl")
include("estimation/bayes.jl")
include("estimation/diagnostics.jl")

include("analytical_solutions/analytical_problem.jl")
include("analytical_solutions/analytical_solution_type.jl")
include("analytical_solutions/standard_models.jl")

include("simulate_methods/utils.jl")
include("simulate_methods/diffeqs.jl")
include("simulate_methods/analytical.jl")

@reexport using .NCA

example_nmtran_data(filename) = joinpath(joinpath(@__DIR__, ".."),"examples/"*filename*".csv")

export Subject, Population, DosageRegimen
export PuMaSModel, init_param, init_randeffs, sample_randeffs
export simobs, pre
export tad, eventnum
export conditional_nll, FIM
export npde, wres, cwres, cwresi, pred, cpred, cpredi, epred, iwres, icwres, icwresi, eiwres
export AIC, BIC, ηshrinkage, ϵshrinkage, ipred, cipred, cipredi
export process_nmtran, example_nmtran_data
export @model, @nca

end # module
