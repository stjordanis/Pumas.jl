module NCA

using Reexport
using GLM
@reexport using DataFrames
using LabelledArrays: LArray
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
  @eval function $f(pop::NCAPopulation, args...; kwargs...)
    labels = ((Symbol(:id, subj.id) for subj in pop)...,)
    params = map(pop->$f(pop, args...; kwargs...), pop)
    return LArray{labels}(params)
  end
end

end
