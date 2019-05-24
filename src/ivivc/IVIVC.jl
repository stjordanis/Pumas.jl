module IVIVC

using Reexport
using RecipesBase
using OrdinaryDiffEq
using CSV, DataFrames

@reexport using DataInterpolations, Optim, ..NCA

abstract type Ivivc end

include("type.jl")
include("data_parsing.jl")
include("models.jl")
include("deconvo_methods.jl")
include("stats.jl")
include("plot_rec.jl")
include("utils.jl")
include("main.jl")

export VitroSubject, VitroPopulation
export VivoSubject, VivoPopulation
export read_vitro, read_vivo
export emax, emax_ng, weibull, double_weibull, makoid
export vitro_model, get_avail_models
export vivo_model, get_avail_vivo_models
export calc_input_rate, wagner_nelson, do_ivivc
export loglikelihood, nullloglikelihood, dof, nobs, deviance, mss,
       rss, aic, aicc, bic, r2
end # module
