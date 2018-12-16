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

@inline function cleanblq(conc′, time′; llq=nothing, concblq=nothing, missingconc=nothing, check=true, kwargs...)
  conc, time = cleanmissingconc(conc′, time′; missingconc=missingconc, check=check)
  isempty(conc) && return conc, time
  llq === nothing && (llq = zero(eltype(conc)))
  concblq === nothing && (concblq = :drop)
  concblq === :keep && return conc, time
  firstidx = ctfirst_idx(conc, time, llq=llq, check=false)
  if firstidx == -1
    # All measurements are BLQ; so apply the "first" BLQ rule to everyting.
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

normalize(x::Number, d::NCADose) = x/d.amt

Base.@propagate_inbounds function ithdoseidxs(time, dose, i::Integer)
  m = length(dose)
  @boundscheck begin
    1 <= i <= m || throw(BoundsError(dose, i))
  end
  idx1 = searchsortedfirst(time, dose[i].time)
  idxs = if i === m
    idx1:length(time)
  else
    idx2 = searchsortedfirst(time, dose[i+1].time)-1
    idx1:idx2
  end
  return idxs
end

Base.@propagate_inbounds function subject_at_ithdose(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I},
                                                     i::Integer) where {C,T,AUC,AUMC,D<:AbstractArray,Z,F,N,I}
  conc = nca.conc[i]
  time = nca.time[i]
  maxidx = nca.maxidx[i]
  lastidx = nca.lastidx[i]
  dose = nca.dose[i]
  # TODO: caching
  #lambdaz = nca.lambdaz[i]
  #r2 = nca.r2[i]
  #point = nca.point[i]
  #NCASubject{typeof(conc),typeof(time),eltype(AUC),eltype(AUMC),typeof(dose),typeof(lambdaz),typeof(r2),N,eltype(I)}(
  NCASubject{typeof(conc),typeof(time),eltype(AUC),eltype(AUMC),typeof(dose),eltype(Z),eltype(F),N,eltype(I)}(
               nca.id,
               conc,    time,
               maxidx,  lastidx,
               dose,
               nothing, nca.llq, nothing, nothing, nothing,
               nothing, nothing)
end

#function remakesubject(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I}, conc, time, dose) where {C,T,AUC,AUMC,D,Z,F,N,I}
#  _, maxidx = conc_maximum(conc, eachindex(conc))
#  lastidx = ctlast_idx(conc, time; llq=nca.llq, check=false)
#  NCASubject{typeof(conc),typeof(time),AUC,AUMC,typeof(dose),Z,F,N,I}(
#               nca.id,
#               conc,    time,
#               maxidx,  lastidx,
#               dose,
#               nothing, nca.llq, nothing, nothing,
#               nothing, nothing,
#               nothing, nothing)
#end
