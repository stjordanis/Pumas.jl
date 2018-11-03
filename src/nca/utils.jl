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
  length(conc) != length(time) && throw(ArgumentError("Concentration and time must be the same length"))
  return
end

function cleanmissingconc(conc, time, args...; missingconc=:drop, check=true)
  check && checkconctime(conc, time)
  T = Base.nonmissingtype(eltype(conc))
  if missingconc == :drop
    idxs = .!(ismissing.(conc))
    newconc = similar(conc, T, count(idxs))
    ii = 1
    @inbounds for i in eachindex(idxs)
      idxs[i] && (newconc[ii] = conc[i]; ii+=1)
    end
    return newconc, time[idxs]
  elseif missingconc isa Number
    newconc = similar(conc, T)
    @inbounds for i in eachindex(conc)
      newconc[i] = ismissing(conc[i]) ? missingconc : conc[i]
    end
    return newconc, time
  else
    throw(ArgumentError("missingconc must be a number or :drop"))
  end
end
