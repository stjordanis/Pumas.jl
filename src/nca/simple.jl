"""
  clast(nca::NCAdata)

Calculate `clast`
"""
function clast(nca::NCAdata; kwargs...)
  idx = nca.lastidx
  return idx === -1 ? missing : nca.conc[idx]
end

"""
  tlast(nca::NCAdata)

Calculate `tlast`
"""
function tlast(nca::NCAdata; kwargs...)
  idx = nca.lastidx
  return idx === -1 ? missing : nca.time[idx]
end

# This function uses ``-1`` to denote missing as after checking `conc` is
# strictly great than ``0``.
function ctlast_idx(conc, time; llq=nothing, check=true)
  check && checkconctime(conc, time)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findlast(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

function ctfirst_idx(conc, time; llq=nothing, check=true)
  check && checkconctime(conc, time)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findfirst(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

"""
  tmax(nca::NCAdata; interval=(0.,Inf), kwargs...)

Calculate ``T_{max}_{t_1}^{t_2}``
"""
function tmax(nca::NCAdata; interval=(0.,Inf), kwargs...)
  if interval isa Tuple
    return ctmax(nca; interval=interval, kwargs...)[2]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctmax(nca; interval=i, kwargs...)[2]
    end
    map(f, interval)
  end
end

"""
  cmax(nca::NCAdata; interval=(0.,Inf), kwargs...)

Calculate ``C_{max}_{t_1}^{t_2}``
"""
function cmax(nca::NCAdata; interval=(0.,Inf), kwargs...)
  dose = nca.dose
  if interval isa Tuple
    sol = ctmax(nca; interval=interval, kwargs...)[1]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctmax(nca; interval=i, kwargs...)[1]
    end
    sol = map(f, interval)
  end
  dose === nothing ? sol : map(s->(cmax=s, cmax_dn=s/dose), sol)
end

@inline function ctmax(nca::NCAdata; interval=(0.,Inf), kwargs...)
  conc, time = nca.conc, nca.time
  if interval === (0., Inf)
    val, idx = conc_maximum(conc, eachindex(conc))
    return (cmax=val, tmax=time[idx])
  end
  @assert interval[1] < interval[2] "t0 must be less than t1"
  interval[1] > time[end] && throw(ArgumentError("t0 is longer than observation time"))
  idx1, idx2 = let (lo, hi)=interval
    findfirst(t->t>=lo, time),
    findlast( t->t<=hi, time)
  end
  val, idx = conc_maximum(conc, idx1:idx2)
  return (cmax=val, tmax=time[idx])
end

@inline function conc_maximum(conc, idxs)
  idx = -1
  val = -oneunit(Base.nonmissingtype(eltype(conc)))
  for i in idxs
    if !ismissing(conc[i])
      val < conc[i] && (val = conc[i]; idx=i)
    end
  end
  return val, idx
end

"""
  thalf(nca::NCAdata; kwargs...)

Calculate half life time.
"""
thalf(nca::NCAdata; kwargs...) = log(2)/lambdaz(nca; recompute=false, kwargs...)[1]

"""
  clf(nca::NCAdata; kwargs...)

Calculate total drug clearance divided by the bioavailability (F), which is just the
inverse of the dose normalized ``AUC_0^\\inf``.
"""
function clf(nca::NCAdata{C,T,AUC,AUMC,D,Z,F,N}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N}
  if D === Nothing
    throw(ArgumentError("Dose must be known to compute CLF"))
  end
  inv(auc(nca; kwargs...)[2])
end

"""
  vss(nca::NCAdata; kwargs...)

Calculate apparent volume of distribution at equilibrium for IV bolus doses.
``V_{ss} = AUMC / {AUC_0^\\inf}^2`` for dose normalized `AUMC` and `AUC`.
"""
function vss(nca::NCAdata{C,T,AUC,AUMC,D,Z,F,N}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N}
  if D === Nothing
    throw(ArgumentError("Dose must be known to compute V_ss"))
  end
  aumc(nca; kwargs...)[2] / (auc(nca; kwargs...)[2])^2
end

"""
  vz(nca::NCAdata; kwargs...)

Calculate the volume of distribution during the terminal phase.
``V_z = 1/(AUC_0^\\inf\\lambda_z)`` for dose normalized `AUC`.
"""
function vz(nca::NCAdata{C,T,AUC,AUMC,D,Z,F,N}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N}
  if D === Nothing
    throw(ArgumentError("Dose must be known to compute V_z"))
  end
  inv(auc(nca; kwargs...)[2] * lambdaz(nca; recompute=false, kwargs...)[1])
end
