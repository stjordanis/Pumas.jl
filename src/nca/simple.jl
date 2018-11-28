"""
  ctlast(conc, time; llq=0, check=true) -> (clast, tlast)

Calculate `clast` and `tlast`.
"""
function ctlast(conc, time; kwargs...)
  idx = _ctlast(conc, time; kwargs...)
  idx === -1 && (clast = missing; tlast = missing)
  clast, tlast = conc[idx], time[idx]
  return (clast=clast, tlast=tlast)
end

# This function uses ``-1`` to denote missing as after checking `conc` is
# strictly great than ``0``.
function ctlast_idx(conc, time; llq=nothing, check=true)
  if check
    checkconctime(conc, time)
  end
  llq = zero(eltype(conc))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  all(f, conc) && return -1
  @inbounds idx = findlast(!f, conc)
  return idx
end

"""
  ctmax(conc, time; interval=(0.,Inf), check=true) -> (cmax, tmax)

Calculate ``C_{max}_{t_1}^{t_2}`` and ``T_{max}_{t_1}^{t_2}``
"""
@inline function ctmax(conc, time; interval=(0.,Inf), check=true)
  if interval === (0., Inf)
    val, idx = conc_maximum(conc, eachindex(conc))
    return (cmax=val, tmax=time[idx])
  end
  check && checkconctime(conc, time)
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
  val = -one(Base.nonmissingtype(eltype(conc)))
  for i in idxs
    if !ismissing(conc[i])
      val < conc[i] && (val = conc[i]; idx=i)
    end
  end
  return val, idx
end

function thalf(conc, time; lambdaz=nothing, kwargs...)
  ln2 = log(2)
  lambdaz === nothing ? ln2/lambdaz(conc, time; kwargs...)[1] : ln2/lambdaz
end
