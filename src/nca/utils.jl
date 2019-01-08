const Maybe{T} = Union{Missing, T}

"""
  checkconctime(conc, time=nothing; monotonictime=true)
Verify that the concentration and time are valid

If the concentrations or times are invalid, will provide an error.
Reasons for being invalid are:
  1. `conc` is not a `Number`
  2. `time` is not a `Number`
  3. `time` value is `missing`
  4. `time` is not monotonically increasing
  5. `conc` and `time` are not of same length

Some cases may generate warnings
  1.  A negative concentration is often but not always an error; it will generate a warning.
"""
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
  elseif any(x -> x<zero(x), skipmissing(conc))
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

"""
  cleanmissingconc(conc, time; missingconc=nothing, check=true)

Handle `missing` values in the concentration measurements as requested by the user.

`missing` concentrations (and their associated times) will be removed

#Arguments
- `missingconc`: How to handle `missing` concentrations?  Either 'drop' or a number to impute.
"""
function cleanmissingconc(conc, time; missingconc=nothing, check=true)
  check && checkconctime(conc, time)
  missingconc === nothing && (missingconc = :drop)
  Ec = eltype(conc)
  Tc = Base.nonmissingtype(Ec)
  Et = eltype(time)
  Tt = Base.nonmissingtype(Et)
  n = count(ismissing, conc)
  # fast path
  Tc === Ec && Tt === Et && n == 0 && return conc, time
  len = length(conc)
  if missingconc === :drop
    newconc = similar(conc, Tc, len-n)
    newtime = similar(time, Tt, len-n)
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
    newconc = similar(conc, Tc)
    newtime = Tt === Et ? time : similar(time, Tt)
    @inbounds for i in eachindex(newconc)
      newconc[i] = ismissing(conc[i]) ? missingconc : conc[i]
    end
    return newconc, time
  else
    throw(ArgumentError("missingconc must be a number or :drop"))
  end
end

"""
  cleanblq(conc′, time′; llq=nothing, concblq=nothing, missingconc=nothing, check=true, kwargs...)

Handle BLQ values in the concentration measurements as requested by the user.

`missing` concentrations (and their associated times) will be handled as described in
`cleanmissingconc` before working with the BLQ values.  The method for handling `missing`
concentrations can affect the output of which points are considered BLQ and which are
considered "middle".  Values are considered BLQ if they are 0.

#Arguments
- `conc`: Measured concentrations
- `time`: Time of the concentration measurement
- `...` Additional arguments passed to `cleanmissingconc`

- `concblq`: How to handle a BLQ value that is between above LOQ values?  See details for description.
- `cleanmissingconc`: How to handle NA concentrations.

`concblq` can be set either a scalar indicating what should be done for all BLQ
values or a list with elements named "first", "middle", and "last" each set to a scalar.
If `nothing`, BLQ values will be dropped (:drop)

The meaning of each of the list elements is:

  1. first: Values up to the first non-BLQ value.  Note that if all values are BLQ,
     this includes all values.
  2. middle: Values that are BLQ between the first and last non-BLQ values.
  3. last: Values that are BLQ after the last non-BLQ value

 The valid settings for each are:
   1. "drop" Drop the BLQ values
   2. "keep" Keep the BLQ values
   3. a number Set the BLQ values to that number
"""
@inline function cleanblq(conc′, time′; llq=nothing, concblq=nothing, missingconc=nothing, check=true, kwargs...)
  conc, time = cleanmissingconc(conc′, time′; missingconc=missingconc, check=check)
  isempty(conc) && return conc, time
  llq === nothing && (llq = zero(eltype(conc)))
  concblq === nothing && (concblq = :drop)
  concblq === :keep && return conc, time
  firstidx = ctfirst_idx(conc, time, llq=llq, check=false)
  if firstidx == -1 # if no firstidx is found, i.e., all measurements are BLQ
    # All measurements are BLQ; so apply the "first" BLQ rule to everyting,
    # hence, we take `tfirst` to be the `last(time)`
    tfirst = last(time)
    tlast = tfirst + one(tfirst)
  else
    tfirst = time[firstidx]
    lastidx = ctlast_idx(conc, time, llq=llq, check=false)
    tlast = time[lastidx]
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
      mask = let tlast=tlast, llq=llq
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
  return conc, time
end

@inline normalizedose(x::Number, d::NCADose) = x/d.amt
normalizedose(x::AbstractArray, d::AbstractVector{<:NCADose}) = normalizedose.(x, d)
@inline function normalizedose(x, subj::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P,ID}) where {C,T,AUC,AUMC,D,Z,F,N,I,P,ID}
  D === Nothing && throw(ArgumentError("Dose must be known to compute normalizedosed quantity"))
  return normalizedose(x, subj.dose)
end

Base.@propagate_inbounds function ithdoseidxs(time, dose, i::Integer)
  m = length(dose)
  @boundscheck 1 <= i <= m || throw(BoundsError(dose, i))
  # get the first index of the `i`-th dose interval
  idx1 = searchsortedfirst(time, dose[i].time)
  idxs = if i === m
    idx1:length(time) # the indices of the last dose interval is `idx1:end`
  else
    idx2 = searchsortedfirst(time, dose[i+1].time)-1
    # indices of the `i`-th dose interval is from `idx1` to the first index of
    # `i+1`-th dose minus one
    idx1:idx2
  end
  return idxs
end

Base.@propagate_inbounds function subject_at_ithdose(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P,ID},
                                                     i::Integer) where {C,T,AUC,AUMC,D<:AbstractArray,Z,F,N,I,P,ID}
  m = length(nca.dose)
  @boundscheck 1 <= i <= m || throw(BoundsError(nca.dose, i))
  @inbounds begin
    conc = nca.conc[i]
    time = nca.time[i]
    maxidx = nca.maxidx[i]
    lastidx = nca.lastidx[i]
    dose = nca.dose[i]
    lambdaz = view(nca.lambdaz, i)
    r2 = view(nca.r2, i)
    intercept = view(nca.intercept, i)
    points = view(nca.points, i)
    auc, aumc = view(nca.auc_last, i), view(nca.aumc_last, i)
    return NCASubject(
                 nca.id,
                 conc,    time,                           # NCA measurements
                 maxidx,  lastidx,                        # idx cache
                 dose,                                    # dose
                 lambdaz, nca.llq, r2, intercept, points, # lambdaz related cache
                 auc, aumc                                # AUC related cache
                )
  end
end
