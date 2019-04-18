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
    ismulti = ismultidose(pop)
    if ismulti
      sol′ = map(enumerate(pop)) do (i, subj)
        try
          if $f == mat
            _sol = $f(subj, args...; kwargs...)
            param  = vcat(_sol, fill(missing, length(subj.dose)-1)) # make `f` as long as the other ones
          else
            param = $f(subj, args...; kwargs...)
          end
        catch
          @info "ID $(subj.id) errored"
          rethrow()
        end
      end
      sol = collect(Base.Iterators.flatten(sol′))
    else
      sol = map(subj -> $f(subj, args...; kwargs...), pop)
    end
    df = DataFrame()
    if label
      _repeat(x, n) = n == 1 ? x : repeat(x, inner=n)
      firstsubj = first(pop)
      ndose = ismulti ? length(firstsubj.dose) : 1
      id′ = map(subj->subj.id, pop)
      df.id = _repeat(id′, ndose)
      ismulti && (df.occasion = repeat(1:ndose, outer=length(pop)))
      if firstsubj.group !== nothing
        ngroup = firstsubj.group isa AbstractArray ? length(firstsubj.group) : 1
        if ngroup == 1
          grouplabel = Symbol(firstsubj.group.first)
          groupnames = map(subj->subj.group.second, pop)
          setproperty!(df, grouplabel, _repeat(groupnames, ndose))
        else # multi-group
          for i in 1:ngroup
            grouplabel = Symbol(firstsubj.group[i].first)
            groupnames = map(subj->subj.group[i].second, pop)
            setproperty!(df, grouplabel, _repeat(groupnames, ndose))
          end
        end
      end
    end
    df.$f = sol
    return df
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
for f in (:c0, :clast, :tlast, :cmax, :tmax, :cmin, :tmin, :_auc, :tlag, :mrt, :fluctation,
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
