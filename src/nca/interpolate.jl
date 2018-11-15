function interpextrapconc(conc, time, timeout; lambdaz=nothing,
                          clast=nothing, interpmethod=nothing,
                          extrapmethod=:AUCinf, concblq=nothing,
                          missingconc=:drop, check=true)
  check && checkconctime(conc, time) # TODO: blq
  clast, tlast = ctlast(conc, time, check=false)
  conc, time = cleanmissingconc(conc, time, missingcon=missingcon, check=false)
  isempty(timeout) && throw(ArgumentError("timeout must be a vector with at least one element"))
  out = timeout isa AbstractArray ? fill!(similar(timeout), 0) : zero(timeout)
  for i in eachindex(out)
    if ismissing(out[i])
      @warn warning("Interpolation/extrapolation time is missing at the $(i)th index")
    elseif timeout[i] <= tlast
      _out = interpolateconc(conc, time, timeout[i], interpmethod=interpmethod, extrapmethod=extrapmethod,
                      concblq=concblq, missingcon=missingconc, check=false)
      out isa AbstractArray ? (out[i] = _out) : (out = _out)
    else
      _out = interpolateconc(conc, time, timeout[i], interpmethod=interpmethod, extrapmethod=extrapmethod,
                      concblq=concblq, missingcon=missingconc, check=false)
      out isa AbstractArray ? (out[i] = _out) : (out = _out)
    end
  end
  return out
end
