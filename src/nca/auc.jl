"""
    auclinear(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the curve (AUC) in an interval by linear trapezoidal rule.
"""
@inline function auclinear(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
  return ((C₁ + C₂) * Δt) / 2
end

"""
    auclog(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the curve (AUC) in an interval by log-linear trapezoidal
rule. If log-linear trapezoidal is not appropriate for the given data, it will
fall back to the linear trapezoidal rule.
"""
@inline function auclog(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
  # we know that C₂ and C₁ must be non-negative, so we need to check C₁=0,
  # C₂=0, or C₁ = C₂, and fall back to the linear trapezoidal rule.
  C₁ == C₂ || C₁ == zero(C₁) || C₂ == zero(C₂) && return auclinear(C₁, C₂, t₁, t₂)
  ΔC = C₂ - C₁
  lg = log(C₂/C₁)
  return (Δt * ΔC) / lg
end

"""
    aucinf(clast, tlast, lambdaz)

Extrapolate concentration to the infinite.
"""
@inline aucinf(clast, tlast, lambdaz) = clast/lambdaz

"""
    aumclinear(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the first moment of the concentration (AUMC) in an interval
by linear trapezoidal rule.
"""
@inline function aumclinear(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  return auclinear(C₁*t₁, C₂*t₂, t₁, t₂)
end

"""
    aumclog(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the first moment of the concentration (AUMC) in an interval
by log-linear trapezoidal rule. If log-linear trapezoidal is not appropriate
for the given data, it will fall back to the linear trapezoidal rule.
"""
@inline function aumclog(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
  # we know that C₂ and C₁ must be non-negative, so we need to check C₁=0,
  # C₂=0, or C₁ = C₂, and fall back to the linear trapezoidal rule.
  C₁ == C₂ || C₁ == zero(C₁) || C₂ == zero(C₂) && return aumclinear(C₁, C₂, t₁, t₂)
  lg = log(C₂/C₁)
  return Δt*(t₂*C₂ - t₁*C₁)/lg - Δt^2*(C₂-C₁)/lg^2
end

"""
    aumcinf(clast, tlast, lambdaz)

Extrapolate the first moment to the infinite.
"""
@inline aumcinf(clast, tlast, lambdaz) = clast*tlast/lambdaz + clast/lambdaz^2

@inline function _auc(conc, time; interval=(0,Inf), clast=nothing, lambdaz=nothing,
              auctype::Symbol=:AUCall, method=nothing, concblq=nothing, # TODO: concblq
              missingconc=:drop, check=true, linear, log, inf, idxs=nothing)
  if check
    checkconctime(conc, time)
    conc, time = cleanmissingconc(conc, time)
  end
  interval[1] >= interval[2] && throw(ArgumentError("The AUC interval must be increasing, got interval=$interval"))
  auctype === :AUCinf && isfinite(interval[2]) && @warn "Requesting AUCinf when the end of the interval is not Inf"
  lo, hi = interval
  lo < first(time) && @warn "Requesting an AUC range starting $lo before the first measurement $(first(time)) is not allowed"
  lo > last(time) && @warn "AUC start time $lo is after the maximum observed time $(last(time))"
  lambdaz === nothing && (lambdaz = find_lambdaz(conc, time, check=false, idxs=idxs)[1])
  # TODO: handle `auclinlog` with IV bolus.
  #
  # `C0` is the concentration at time zero if the route of administration is IV
  # (intranvenous). If concentration at time zero is not measured after the IV
  # bolus, a linear back-extrapolation is done to get the intercept which
  # represents concentration at time zero (`C0`)
  #
  concstart = interpextrapconc(conc, time, lo, lambdaz=lambdaz, interpmethod=method, extrapmethod=auctype, check=false)
  idx1, idx2 = let lo = lo, hi = hi
    (findfirst(x->x>=lo, time),
     findlast(x->x<=hi, time))
  end
  # auc of the bounary intervals
  __auc = linear(concstart, conc[idx1], lo, time[idx1])
  #__auc = conc[idx1]
  auc::typeof(__auc) = __auc
  if conc[idx1] != concstart
    auc = linear(concstart, conc[idx], lo, time[idx1])
    int_idxs = idx1+1:idx2-1
  else
    auc = linear(conc[idx1], conc[idx1+1], time[idx1], time[idx1+1])
    int_idxs = idx1+1:idx2-1
  end
  if isfinite(hi)
    concend = interpextrapconc(conc, time, hi, lambdaz=lambdaz, interpmethod=method, extrapmethod=auctype, check=false)
    if conc[idx2] != concend
      if method === :linear || (method === :linuplogdown && concend ≥ conc[idx2]) # increasing
        auc += linear(conc[idx2], concend, time[idx2], hi)
      else
        auc += log(conc[idx2], concend, time[idx2], hi)
      end
    end
  end

  _clast, tlast = ctlast(conc, time, check=false)
  clast == nothing && (clast = _clast)
  if clast == -one(eltype(conc))
    return all(isequal(0), conc) ? (zero(auc), zero(auc), lambdaz) : error("Unknown error with missing `tlast` but non-BLQ concentrations")
  end

  # Compute the AUxC
  # Include the first point after `tlast` if it exists and we are computing AUCall
  # do not support :AUCall for now
  # auctype === :AUCall && tlast > hi && (idx2 = idx2 + 1)
  if method === :linear
    for i in int_idxs
      auc += linear(conc[i], conc[i+1], time[i], time[i+1])
    end
  elseif method === :linuplogdown
    for i in int_idxs
      if conc[i] ≥ conc[i-1] # increasing
        auc += linear(conc[i], conc[i+1], time[i], time[i+1])
      else
        auc += log(conc[i], conc[i+1], time[i], time[i+1])
      end
    end
  else
    throw(ArgumentError("method must be either :linear or :linuplogdown"))
  end
  aucinf2 = inf(clast, tlast, lambdaz)
  return auc, auc+aucinf2, lambdaz
end

"""
    auc(concentration::Vector{<:Real}, time::Vector{<:Real}; method::Symbol, kwargs...)

Compute area under the curve (AUC) by linear trapezoidal rule `(method =
:linear)` or by log-linear trapezoidal rule `(method = :linuplogdown)`. It
returns a tuple in the form of `(AUC_0_last, AUC_0_inf, λz)`.
"""
auc(conc, time; kwargs...) = _auc(conc, time, linear=auclinear, log=auclog, inf=aucinf; kwargs...)

"""
    aumc(concentration::Vector{<:Real}, time::Vector{<:Real}; method::Symbol, kwargs...)

Compute area under the first moment of the concentration (AUMC) by linear
trapezoidal rule `(method = :linear)` or by log-linear trapezoidal rule
`(method = :linuplogdown)`. It returns a tuple in the form of `(AUC_0_last,
AUC_0_inf, λz)`.
"""
aumc(conc, time; kwargs...) = _auc(conc, time, linear=aumclinear, log=aumclog, inf=aumcinf; kwargs...)

fitlog(x, y) = lm(hcat(fill!(similar(x), 1), x), log.(y[y.!=0]))

"""
  find_lambdaz(conc, time; threshold=10, check=true, idxs=nothing,
  missingconc=:drop) -> (lambdaz, points, r2)

Calculate ``λz``, ``r^2``, and the number of data points from the profile used
in the determination of ``λz``.
"""
function find_lambdaz(conc, time; threshold=10, check=true, idxs=nothing, missingconc=:drop) #TODO: better name
  if check
    checkconctime(conc, time)
    conc, time = cleanmissingconc(conc, time, missingconc=missingconc, check=false)
  end
  maxr2 = 0.0
  λ = 0.0
  points = 2
  outlier = false
  cmax, cmaxidx = findmax(conc)
  n = length(time)
  if idxs === nothing
    m = min(n-1, threshold, n-cmaxidx-1)
    m < 2 && throw(ArgumentError("lambdaz must be calculated from at least three data points after Cmax"))
    for i in 2:m
      x = time[end-i:end]
      y = conc[end-i:end]
      model = fitlog(x, y)
      r2 = r²(model)
      if r2 > maxr2
        maxr2 = r2
        λ = coef(model)[2]
        points = i+1
      end
    end
  else
    points = length(idxs)
    points < 3 && throw(ArgumentError("lambdaz must be calculated from at least three data points"))
    x = time[idxs]
    y = conc[idxs]
    model = fitlog(x, y)
    λ = coef(model)[2]
    r2 = r²(model)
  end
  if λ ≥ 0
    outlier = true
    @warn "The estimated slope is not negative, got $λ"
  end
  return (lambdaz=-λ, points=points, r2=r2)
end
