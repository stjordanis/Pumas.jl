module IVIVC

using DataInterpolations
using Optim
using RecipesBase
using Reexport

include("type.jl")
include("data_parsing.jl")
include("empirical_models.jl")
include("model_caches.jl")
include("stats.jl")
include("plot_rec.jl")
include("utils.jl")

export VitroSubject, VitroPopulation
export VivoSubject, VivoPopulation
export read_vitro, read_vivo
export model_fit
export EmaxModel, WeibullModel, DoubleWeibullModel, MakoidModel
export likelihood, loglikelihood, nullloglikelihood, dof, nobs, deviance, mss,
       rss, aic, aicc, bic, r2
end # module
