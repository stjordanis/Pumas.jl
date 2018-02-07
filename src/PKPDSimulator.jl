__precompile__()

module PKPDSimulator

using DiffEqBase, DiffEqMonteCarlo, Distributions, Reexport, NamedTuples,
      StaticArrays, DiffEqJump, PDMats

@reexport using OrdinaryDiffEq


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
include("simulate_methods/model_type.jl")
include("simulate_methods/error_models.jl")
include("simulate_methods/diffeqs.jl")
include("simulate_methods/analytical.jl")


export Subject, Population, process_data, build_dataset

export simulate, ith_subject_cb
export PKPDModel, FullModel, ErrorModel, Independent

end # module
