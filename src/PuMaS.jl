module PuMaS

using DiffEqBase, DiffEqMonteCarlo, Reexport, DataFrames,
      StaticArrays, DiffEqJump, Distributed, LabelledArrays, GLM,
      TreeViews

@reexport using OrdinaryDiffEq, DelayDiffEq, Unitful
@reexport using Distributions, PDMats

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

export Subject, Population, process_nmtran, DosageRegimen
export PKPDModel, init_param, init_random, rand_random,
       simobs, likelihood, pre, simpost,
       AUC, AUMC

end # module
