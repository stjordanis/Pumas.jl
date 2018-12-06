module NCA

using Reexport
using GLM
@reexport using DataFrames
using Pkg, Dates, Printf

include("utils.jl")
include("type.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export NCAData, showunits
export auc, aumc, lambdaz, auc_extrap_percent, aumc_extrap_percent,
       clast, tlast, cmax, tmax, thalf, clf, vss, vz
export NCAReport

for f in (:lambdaz, :cmax, :tmax, :clast, :tlast, :thalf, :clf, :vss, :vz,
          :interpextrapconc, :auc, :aumc, :auc_extrap_percent, :aumc_extrap_percent)
  @eval $f(conc, time, args...; kwargs...) = $f(NCAData(conc, time; kwargs...), args...; kwargs...)
end

end
