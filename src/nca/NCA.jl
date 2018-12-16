module NCA

using Reexport
using GLM
@reexport using DataFrames
using Pkg, Dates, Printf
import ..PuMaS: Formulation, IV, EV

include("type.jl")
include("data_parsing.jl")
include("utils.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export NCASubject, NCAPopulation, NCADose, showunits
export parse_ncadata
export auc, aumc, lambdaz, auc_extrap_percent, aumc_extrap_percent,
       clast, clast, tlast, cmax, tmax, thalf, cl, clf, vss, vz,
       bioavailability, tlag, mrt, mat
export NCAReport

for f in (:lambdaz, :cmax, :tmax, :clast, :clast, :tlast, :thalf, :cl, :clf, :vss, :vz,
          :interpextrapconc, :auc, :aumc, :auc_extrap_percent, :aumc_extrap_percent,
          :bioavailability, :tlag, :mrt, :mat)
  @eval $f(conc, time, args...; kwargs...) = $f(NCASubject(conc, time; kwargs...), args...; kwargs...)
  @eval $f(n::NCAPopulation, args...; kwargs...) = map(n->$f(n, args...; kwargs...), n)
end

end
