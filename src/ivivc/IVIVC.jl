module IVIVC

using Reexport
using RecipesBase

@reexport  using DataInterpolations, Optim

abstract type Ivivc end

include("type.jl")
include("data_parsing.jl")
include("models.jl")
include("stats.jl")
include("plot_rec.jl")
include("utils.jl")

export VitroSubject, VitroPopulation
export VivoSubject, VivoPopulation
export read_vitro, read_vivo
export emax, emax_ng, weibull, double_weibull, makoid
export vitro_model, get_avail_models
export loglikelihood, nullloglikelihood, dof, nobs, deviance, mss,
       rss, aic, aicc, bic, r2
end # module
