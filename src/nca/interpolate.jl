function interpextrapconc(nca::NCASubject, timeout; concorigin=nothing, interpmethod=nothing, kwargs...)
  conc, time = nca.conc, nca.time
  _tlast = tlast(nca)
  isempty(timeout) && throw(ArgumentError("timeout must be a vector with at least one element"))
  out = timeout isa AbstractArray ? fill!(similar(timeout), 0) : zero(timeout)
  for i in eachindex(out)
    if ismissing(out[i])
      @warn warning("Interpolation/extrapolation time is missing at the $(i)th index")
    elseif timeout[i] <= _tlast
      _out = interpolateconc(nca, timeout[i], interpmethod=interpmethod, concorigin=concorigin)
      out isa AbstractArray ? (out[i] = _out) : (out = _out)
    else
      _out = extrapolateconc(nca, timeout[i])
      out isa AbstractArray ? (out[i] = _out) : (out = _out)
    end
  end
  return out
end

function interpolateconc(nca::NCASubject, timeout::Number; interpmethod, concorigin=nothing, kwargs...)
  conc, time = nca.conc, nca.time
  concorigin === nothing && (concorigin=zero(eltype(conc)))
  len = length(time)
  !(concorigin isa Number) && !(concorigin isa Bool) && throw(ArgumentError("concorigin must be a scalar"))
  _tlast = tlast(nca)
  !(interpmethod in (:linear, :linuplogdown, :linlog)) && throw(ArgumentError("Interpolation method must be :linear, :linuplogdown or :linlog"))
  if timeout < first(time)
    return concorigin
  elseif timeout > _tlast
    throw(ArgumentError("interpolateconc can only works through Tlast, please use interpextrapconc to combine both interpolation and extrapolation"))
  elseif (idx2=searchsortedfirst(time, timeout)) != len+1 && time[idx2] == timeout # if there is an exact time match
    return conc[idx2]
  else
    idx1 = idx2 - 1
    time1 = time[idx1]; time2 = time[idx2]
    conc1 = conc[idx1]; conc2 = conc[idx2]
    m = choosescheme(conc1, conc2, time1, time2, idx1, nca.maxidx, interpmethod)
    if m === Linear
      return conc1+abs(timeout-time1)/(time2-time1)*(conc2-conc1)
    else
      return exp(log(conc1)+(timeout-time1)/(time2-time1)*(log(conc2)-log(conc1)))
    end
  end
end

function extrapolateconc(nca::NCASubject, timeout::Number; kwargs...)
  conc, time = nca.conc, nca.time
  λz = lambdaz(nca; recompute=false, kwargs...)[1]
  _tlast = tlast(nca)
  _clast = clast(nca)
  #!(extrapmethod === :AUCinf) &&
    #throw(ArgumentError("extrapmethod must be one of AUCinf"))
  if timeout <= _tlast
    throw(ArgumentError("extrapolateconc can only work beyond Tlast, please use interpextrapconc to combine both interpolation and extrapolation"))
  else
    #if extrapmethod === :AUCinf
      # If AUCinf is requested, extrapolate using the half-life
    return _clast*exp(-λz*(timeout - _tlast))
    #elseif extrapmethod === :AUCall && tlast == (maxtime=maximum(time))
    #  # If AUCall is requested and there are no BLQ at the end, we are already
    #  # certain that we are after Tlast, so the answer is 0.
    #  return oneunit(eltype(conc))*false
    #elseif extrapmethod === :AUCall
    #  # If the last non-missing concentration is below the limit of
    #  # quantification, extrapolate with the triangle method of AUCall.
    #  previdx = findlast(t->t<=timeout, time)
    #  timeprev, concprev = time[previdx], concprev[previdx]
    #  if iszero(concprev)
    #    return oneunit(eltype(conc))*false
    #  else
    #    # If we are not already BLQ, then we have confirmed that we are in the
    #    # triangle extrapolation region and need to draw a line.
    #    #if timeprev != maxtime
    #      nextidx = findfirst(t->t=>timeout, time)
    #      timenext, concnext = time[nextidx], conc[nextidx]
    #    #end
    #    return (timeout - timeprev)/(timenext - timeprev)*concprev
    #  end
    #else
    #  error("Invalid extrap.method caught too late (seeing this error indicates a software bug)")
    #  return nothing
    #end
  end
end
