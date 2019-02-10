module NCA

using Reexport
using GLM
using DataFrames
using RecipesBase
using Pkg, Dates, Printf
import ..PuMaS: Formulation, IV, EV

include("type.jl")
include("data_parsing.jl")
include("utils.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export DataFrame

export NCASubject, NCAPopulation, NCADose, showunits
export parse_ncadata
#export auc, aumc, lambdaz, auc_extrap_percent, aumc_extrap_percent,
#       clast, tlast, cmax, tmax, cmin, c0, tmin, thalf, cl, clf, vss, vz,
#       bioav, tlag, mrt, mat, tau, cavg, fluctation, accumulationindex,
#       swing
export NCAReport
export normalizedose

for f in (:lambdaz, :lambdazr2, :lambdazadj2, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst,
          :cmax, :tmax, :cmin, :c0, :tmin, :clast, :tlast, :thalf, :cl, :clf, :vss, :vz,
          :interpextrapconc, :auc, :aumc, :auc_extrap_percent, :aumc_extrap_percent,
          :bioav, :tlag, :mrt, :mat, :tau, :cavg, :fluctation, :accumulationindex,
          :swing)
  @eval $f(conc, time, args...; kwargs...) = $f(NCASubject(conc, time; kwargs...), args...; kwargs...) # f(conc, time) interface
  @eval function $f(pop::NCAPopulation, args...; kwargs...) # NCAPopulation handling
    res = map(pop) do subj
      sol = $f(subj, args...; kwargs...)
      sol isa NamedTuple ? (id=subj.id, $f(subj, args...; kwargs...)...,) : (id=subj.id, $f=$f(subj, args...; kwargs...))
    end
    return DataFrame(res)
  end
end

# Multiple dosing handling
for f in (:clast, :tlast, :cmax, :tmax, :cmin, :tmin, :_auc, :tlag, :mrt, :fluctation,
          :cavg, :tau, :accumulationindex, :swing,
          :lambdaz, :lambdazr2, :lambdazadj2, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst)
  @eval function $f(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P,ID}, args...; kwargs...) where {C,T,AUC,AUMC,D<:AbstractArray,Z,F,N,I,P,ID}
    obj = map(eachindex(nca.dose)) do i
      subj = subject_at_ithdose(nca, i)
      $f(subj, args...; kwargs...)
    end
  end
end

end
