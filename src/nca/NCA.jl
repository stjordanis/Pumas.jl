module NCA

using Reexport
using GLM
using DataFrames
using RecipesBase
using Pkg, Dates, Printf, LinearAlgebra

include("type.jl")
include("data_parsing.jl")
include("utils.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export DataFrame

export NCASubject, NCAPopulation, NCADose, showunits
export parse_ncadata, add_ii!
#export auc, aumc, lambdaz, auc_extrap_percent, aumc_extrap_percent,
#       clast, tlast, cmax, tmax, cmin, c0, tmin, thalf, cl, clf, vss, vz,
#       bioav, tlag, mrt, mat, tau, cavg, fluctation, accumulationindex,
#       swing, superposition
export NCAReport
export normalizedose

for f in (:lambdaz, :lambdazr2, :lambdazadjr2, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst,
          :cmax, :tmax, :cmin, :c0, :tmin, :clast, :tlast, :thalf, :cl, :clf, :vss, :vz,
          :interpextrapconc, :auc, :auclast, :auctau, :aumc, :aumclast, :aumctau, :auc_extrap_percent, :aumc_extrap_percent,
          :bioav, :tlag, :mrt, :mat, :tau, :cavg, :fluctation, :accumulationindex,
          :swing)
  @eval $f(conc, time, args...; kwargs...) = $f(NCASubject(conc, time; kwargs...), args...; kwargs...) # f(conc, time) interface
  @eval function $f(pop::NCAPopulation, args...; label=true, kwargs...) # NCAPopulation handling
    if ismultidose(pop)
      sol = map(enumerate(pop)) do (i, subj)
        try
          if $f in (mat, c0)
            _sol = $f(subj, args...; kwargs...)
            sol  = vcat(_sol, fill(missing, length(subj.dose)-1)) # make `f` as long as the other ones
          else
            sol = $f(subj, args...; kwargs...)
          end
        catch
          @info "ID $(subj.id) errored"
          rethrow()
        end
        label ? subj.group === nothing ? DataFrame(id=subj.id, occasion=eachindex(sol), $f=sol) : DataFrame(id=subj.id, occasion=eachindex(sol), group=subj.group, $f=sol) : DataFrame($f=sol)
      end
    else
      sol = map(pop) do subj
        label ? subj.group === nothing ? DataFrame(id=subj.id, $f=$f(subj, args...; kwargs...)) : DataFrame(id=subj.id, group=subj.group, $f=$f(subj, args...; kwargs...)) : DataFrame($f=$f(subj, args...; kwargs...))
      end
    end
    return vcat(sol...) # splat is faster than `reduce(vcat, sol)`
  end
end

# special handling for superposition
superposition(conc, time, args...; kwargs...) = superposition(NCASubject(conc, time; kwargs...), args...; kwargs...) # f(conc, time) interface
function superposition(pop::NCAPopulation, args...; kwargs...) # NCAPopulation handling
  sol = map(pop) do subj
    res = superposition(subj, args...; kwargs...)
    DataFrame(id=subj.id, conc=res[1], time=res[2])
  end
  return vcat(sol...) # splat is faster than `reduce(vcat, sol)`
end

# add `tau`
# Multiple dosing handling
for f in (:clast, :tlast, :cmax, :tmax, :cmin, :tmin, :_auc, :tlag, :mrt, :fluctation,
          :cavg, :tau, :auctau, :aumctau, :accumulationindex, :swing,
          :lambdaz, :lambdazr2, :lambdazadjr2, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst)
  @eval function $f(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,II}, args...; kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D<:AbstractArray,Z,F,N,I,P,ID,G,II}
    obj = map(eachindex(nca.dose)) do i
      local subj
      try
        subj = subject_at_ithdose(nca, i)
      catch e
        @info "Errored at $(i)th occasion"
        rethrow()
      end
      $f(subj, args...; kwargs...)
    end
  end
end

end
