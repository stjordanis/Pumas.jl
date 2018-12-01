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
tmax(nca::NCAdata; kwargs...) = ctmax(nca; kwargs...)[2]

"""
  cmax(nca::NCAdata; interval=(0.,Inf), kwargs...)

Calculate ``C_{max}_{t_1}^{t_2}``
"""
function cmax(nca::NCAdata; kwargs...)
  dose = nca.dose
  sol = ctmax(nca; kwargs...)[1]
  dose === nothing ? sol : (sol, sol/dose)
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
  val = -one(Base.nonmissingtype(eltype(conc)))
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
