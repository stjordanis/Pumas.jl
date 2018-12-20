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
       clast, tlast, cmax, tmax, cmin, tmin, thalf, cl, clf, vss, vz,
       bioav, tlag, mrt, mat, tau, cavg, fluctation, accumulationindex,
       swing
export NCAReport

for f in (:lambdaz, :cmax, :tmax, :cmin, :tmin, :clast, :tlast, :thalf, :cl, :clf, :vss, :vz,
          :interpextrapconc, :auc, :aumc, :auc_extrap_percent, :aumc_extrap_percent,
          :bioav, :tlag, :mrt, :mat, :tau, :cavg, :fluctation, :accumulationindex,
          :swing)
  @eval $f(conc, time, args...; kwargs...) = $f(NCASubject(conc, time; kwargs...), args...; kwargs...)
  @eval function $f(pop::NCAPopulation, args...; prefix=true, kwargs...)
    if prefix
      res = map(pop) do subj
        sol = $f(subj, args...; kwargs...)
        sol isa NamedTuple ? (id=subj.id, $f(subj, args...; kwargs...)...,) : (id=subj.id, $f=$f(subj, args...; kwargs...))
      end
    else
      res = map(pop) do subj
        $f(subj, args...; kwargs...)
      end
    end
    return res
  end
end

end
