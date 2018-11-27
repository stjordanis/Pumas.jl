const Maybe{T} = Union{Missing, T}

function checkconctime(conc, time=nothing; monotonictime=true)
  # check conc
  conc == nothing && return
  E = eltype(conc)
  isallmissing = all(ismissing, conc)
  if isempty(conc)
    @warn "No concentration data given"
  elseif !(E <: Maybe{Number} && conc isa AbstractArray) || E <: Maybe{Bool} && !isallmissing
    throw(ArgumentError("Concentration data must be numeric and an array"))
  elseif isallmissing
    @warn "All concentration data is missing"
  elseif any(x -> x<0, skipmissing(conc))
    @warn "Negative concentrations found"
  end
  # check time
  time == nothing && return
  T = eltype(time)
  if isempty(time)
    @warn "No time data given"
  elseif any(ismissing, time)
    throw(ArgumentError("Time may not be missing"))
  elseif !(T <: Maybe{Number} && time isa AbstractArray)
    throw(ArgumentError("Time data must be numeric and an array"))
  end
  if monotonictime
    !issorted(time, lt=≤) && throw(ArgumentError("Time must be monotonically increasing"))
  end
  # check both
  # TODO: https://github.com/UMCTM/PuMaS.jl/issues/153
  length(conc) != length(time) && throw(ArgumentError("Concentration and time must be the same length"))
  return
end

function cleanmissingconc(conc, time, args...; missingconc=nothing, check=true)
  check && checkconctime(conc, time)
  missingconc === nothing && (missingconc = :drop)
  E = eltype(conc)
  T = Base.nonmissingtype(E)
  n = count(ismissing, conc)
  # fast path
  n == 0 && return conc, time
  len = length(conc)
  if missingconc === :drop
    newconc = similar(conc, T, len-n)
    newtime = similar(time, len-n)
    ii = 1
    @inbounds for i in eachindex(conc)
      if !ismissing(conc[i])
        newconc[ii] = conc[i]
        newtime[ii] = time[i]
        ii+=1
      end
    end
    return newconc, newtime
  elseif missingconc isa Number
    newconc = similar(conc, T)
    @inbounds for i in eachindex(newconc)
      newconc[i] = ismissing(conc[i]) ? missingconc : conc[i]
    end
    return newconc, time
  else
    throw(ArgumentError("missingconc must be a number or :drop"))
  end
end

function cleanblq(conc′, time′; llq=nothing, concblq=nothing, missingconc=nothing, check=true)
  conc, time = cleanmissingconc(conc′, time′; missingconc=missingconc, check=check)
  llq === nothing && (llq = zero(eltype(conc)))
  concblq === nothing && (concblq = :keep)
  tfirst = first(time)
  if ismissing(tfirst) || (tlast = _ctlast(conc, time)[2]) == -one(tlast)
    # All measurements are BLQ; so apply the "first" BLQ rule to everyting.
    tfirst = last(time)
    tlast = tfirst + one(tfirst)
  end
  for n in (:first, :middle, :last)
    if n === :first
      mask = let tfirst=tfirst, llq=llq
        f = (c, t)-> t ≤ tfirst && c ≤ llq
        f.(conc, time)
      end
    elseif n === :middle
      mask = let tfirst=tfirst, tlast=tlast, llq=llq
        f = (c,t)-> tfirst < t < tlast && c ≤ llq
        f.(conc, time)
      end
    else # :last
      mask = let tlast=tlast
        f = (c,t) -> tlast <= t && c ≤ llq
        f.(conc, time)
      end
    end
    rule = concblq isa Dict ? concblq[n] : concblq
    if rule === :keep
      # do nothing
    elseif rule === :drop
      conc = conc[.!mask]
      time = time[.!mask]
    elseif rule isa Number
      conc === conc′ && (conc = deepcopy(conc)) # if it aliases with the original array, then copy
      conc[mask] .= rule
    else
      throw(ArgumentError("Unknown BLQ rule: $(repr(rule))"))
    end
  end
  conc, time
end

"""
  ctlast(conc, time; interval=(0.,Inf), check=true) -> (clast, tlast)

Calculate `clast` and `tlast`.
"""
function ctlast(conc, time; check=true)
  clast, tlast = _ctlast(conc, time, check=check)
  clast == -one(eltype(conc)) && (clast = missing; tlast = missing)
  return (clast=clast, tlast=tlast)
end

# This function uses ``-1`` to denote missing as after checking `conc` is
# strictly great than ``0``.
function _ctlast(conc, time; check=true)
  if check
    checkconctime(conc, time)
  end
  # now we assume the data is checked
  all(x->(ismissing(x) || x==0), conc) && return -one(eltype(conc)), -one(eltype(time))
  @inbounds idx = findlast(x->!(ismissing(x) || x==0), conc)
  return conc[idx], time[idx]
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
