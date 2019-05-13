module IVIVC

using Reexport
using RecipesBase

@reexport  using DataInterpolations, Optim

include("type.jl")
include("data_parsing.jl")
# include("empirical_models.jl")
# include("stats.jl")
# include("plot_rec.jl")
include("utils.jl")

export VitroSubject, VitroPopulation
export VivoSubject, VivoPopulation
export read_vitro, read_vivo
export vitro_model, get_avail_models
# export likelihood, loglikelihood, nullloglikelihood, dof, nobs, deviance, mss,
#        rss, aic, aicc, bic, r2
end # module
