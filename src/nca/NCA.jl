module NCA

using Reexport
using GLM
@reexport using DataFrames

include("type.jl")
include("utils.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export NCAdata, NCAdose, showunits
export parse_ncadata
export auc, aumc, lambdaz, auc_extrap_percent, aumc_extrap_percent,
       clast, tlast, cmax, tmax, thalf, clf, vss, vz

for f in (:lambdaz, :cmax, :tmax, :clast, :tlast, :thalf, :clf, :vss, :vz,
          :interpextrapconc, :auc, :aumc, :auc_extrap_percent, :aumc_extrap_percent)
  @eval $f(conc, time, args...; kwargs...) = $f(NCAdata(conc, time; kwargs...), args...; kwargs...)
end

end
