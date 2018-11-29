function interpextrapconc(nca::NCAdata, timeout;                           clast=nothing, interpmethod=nothing,
                          extrapmethod=nothing, concblq=nothing,
                          missingconc=nothing, check=true,
                          concorigin=nothing, llq=nothing, kwargs...)
  conc, time = nca.conc, nca.time
  extrapmethod === nothing && (extrapmethod=:AUCinf)
  λz = lambdaz(nca; kwargs...)[1]
  tlast = time[nca.lastidx]
  isempty(timeout) && throw(ArgumentError("timeout must be a vector with at least one element"))
  out = timeout isa AbstractArray ? fill!(similar(timeout), 0) : zero(timeout)
  for i in eachindex(out)
    if ismissing(out[i])
      @warn warning("Interpolation/extrapolation time is missing at the $(i)th index")
    elseif timeout[i] <= tlast
      _out = interpolateconc(nca, timeout[i], interpmethod=interpmethod,
                      concblq=concblq, missingconc=missingconc, concorigin=concorigin, check=false)
      out isa AbstractArray ? (out[i] = _out) : (out = _out)
    else
      _out = extrapolateconc(conc, time, timeout[i], extrapmethod=extrapmethod, λz=λz,
                      concblq=concblq, missingconc=missingconc, check=false)
      out isa AbstractArray ? (out[i] = _out) : (out = _out)
    end
  end
  return out
end

function interpolateconc(conc, time, timeout::Number; interpmethod,
                         concblq=nothing, missingconc=nothing, concorigin=nothing,
                         llq=nothing, check=true)
  if check
    conc, time = cleanblq(conc, time, concblq=concblq, llq=llq, missingconc=missingconc)
  end
  concorigin === nothing && (concorigin=zero(eltype(conc)))
  len = length(time)
  !(concorigin isa Number) && !(concorigin isa Bool) && throw(ArgumentError("concorigin must be a scalar"))
  _, tlast = _ctlast(conc, time, llq=llq, check=false)
  !(interpmethod in (:linear, :linuplogdown)) && throw(ArgumentError("Interpolation method must be :linear or :linuplogdown"))
  if timeout < first(time)
    return concorigin
  elseif timeout > tlast
    throw(ArgumentError("interpolateconc can only works through Tlast, please use interpextrapconc to combine both interpolation and extrapolation"))
  elseif (idx2=searchsortedfirst(time, timeout)) != len+1 && time[idx2] == timeout # if there is an exact time match
    return conc[idx2]
  else
    idx1 = idx2 - 1
    time1 = time[idx1]; time2 = time[idx2]
    conc1 = conc[idx1]; conc2 = conc[idx2]
    if interpmethod === :linear || (interpmethod === :linuplogdown &&
      (conc1 <= 0 || conc2 <= 0) ||
        (conc1 <= conc2))
      # Do linear interpolation if:
      #   linear interpolation is selected or
      #   lin up/log down interpolation is selected and
      #     one concentration is 0 or
      #     the concentrations are equal
      return conc1+abs(timeout-time1)/(time2-time1)*(conc2-conc1)
    elseif interpmethod === :linuplogdown
      return exp(log(conc1)+(timeout-time1)/(time2-time1)*(log(conc2)-log(conc1)))
    else
      error("You should never see this error. Please report this as a bug with a reproducible example.")
      return nothing
    end
  end
end

function extrapolateconc(conc, time, timeout::Number; λz=nothing, clast=nothing, extrapmethod=nothing,
                         missingconc=nothing, concblq=nothing, llq=nothing, check=true, kwargs...)
  if check
    checkconctime(conc, time) # TODO: blq
    conc, time = cleanmissingconc(conc, time, missingconc=missingconc, check=false)
  end
  __ctlast = _ctlast(conc, time, llq=llq, check=false)
  clast, tlast = clast == nothing ? __ctlast : (clast, __ctlast[2])
  !(extrapmethod === :AUCinf) &&
    throw(ArgumentError("extrapmethod must be one of AUCinf"))
  if timeout <= tlast
    throw(ArgumentError("extrapolateconc can only work beyond Tlast, please use interpextrapconc to combine both interpolation and extrapolation"))
  else
    if extrapmethod === :AUCinf
      # If AUCinf is requested, extrapolate using the half-life
      return clast*exp(-λz*(timeout - tlast))
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
    end
  end
end
