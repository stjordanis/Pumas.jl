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

function _auc(conc, time; interval=(0,Inf), clast=nothing, lambdaz=nothing,
              auctype::Symbol=:AUClast, method=nothing, concblq=nothing, # TODO: concblq
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
  lambdaz === nothing && (lambdaz = find_lambdaz(conc, time, check=false, idxs=idxs))
  # TODO: handle `auclinlog` with IV bolus.
  #
  # `C0` is the concentration at time zero if the route of administration is IV
  # (intranvenous). If concentration at time zero is not measured after the IV
  # bolus, a linear back-extrapolation is done to get the intercept which
  # represents concentration at time zero (`C0`)
  #
  # TODO: data interpolation/data extrapolation
  #concstart = interpextrapconc(conc, time, lo, lambdaz=lambdaz, interpmethod=method, extrapmethod=auctype, check=false)
  idx1 = findfirst(x->x>=lo, time)
  idx2 = findlast(x->x<=hi, time)
  # auc of the first interval
  auc = linear(conc[idx1], conc[idx1+1], time[idx1], time[idx1+1])
  if isfinite(hi)
    #concend = interpextrapconc(conc, time, hi, lambdaz=lambdaz, interpmethod=method, extrapmethod=auctype, check=false)
  end

  _clast, tlast = _ctlast(conc, time, check=false)
  clast == nothing && (clast = _clast)
  if clast == -one(eltype(conc))
    return all(isequal(0), conc) ? (zero(auc), zero(auc), lambdaz) : error("Unknown error with missing `tlast` but non-BLQ concentrations")
  end

  # Compute the AUxC
  # Include the first point after `tlast` if it exists and we are computing AUCall
  # do not support :AUCall for now
  # auctype === :AUCall && tlast > hi && (idx2 = idx2 + 1)
  if method === :linear
    for i in idx1+1:idx2-1
      auc += linear(conc[i], conc[i+1], time[i], time[i+1])
    end
  elseif method === :log_linear
    for i in idx1+1:idx2-1
      if conc[i] ≥ conc[i-1] # increasing
        auc += linear(conc[i], conc[i+1], time[i], time[i+1])
      else
        auc += log(conc[i], conc[i+1], time[i], time[i+1])
      end
    end
  else
    throw(ArgumentError("method must be either :linear or :log_linear"))
  end
  aucinf2 = inf(clast, tlast, lambdaz)
  return auc, auc+aucinf2, lambdaz
end

"""
    auc(concentration::Vector{<:Real}, time::Vector{<:Real}; method::Symbol, kwargs...)

Compute area under the curve (AUC) by linear trapezoidal rule `(method =
:linear)` or by log-linear trapezoidal rule `(method = :log_linear)`. It
returns a tuple in the form of `(AUC_0_last, AUC_0_inf, λz)`.
"""
auc(conc, time; kwargs...) = _auc(conc, time, linear=auclinear, log=auclog, inf=aucinf; kwargs...)

"""
    aumc(concentration::Vector{<:Real}, time::Vector{<:Real}; method::Symbol, kwargs...)

Compute area under the first moment of the concentration (AUMC) by linear
trapezoidal rule `(method = :linear)` or by log-linear trapezoidal rule
`(method = :log_linear)`. It returns a tuple in the form of `(AUC_0_last,
AUC_0_inf, λz)`.
"""
aumc(conc, time; kwargs...) = _auc(conc, time, linear=aumclinear, log=aumclog, inf=aumcinf; kwargs...)

fitlog(x, y) = lm(hcat(fill!(similar(x), 1), x), log.(y[y.!=0]))

function find_lambdaz(conc::Vector{<:Real}, time::Vector{<:Real}; threshold=10, check=true, idxs=nothing)
  check && checkconctime(conc, time)
  maxrsq = 0.0
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
      rsq = r²(model)
      if rsq > maxrsq
        maxrsq = rsq
        λ = coef(model)[2]
        points = i+1
      end
    end
  else
    length(idxs) < 3 && throw(ArgumentError("lambdaz must be calculated from at least three data points"))
    x = time[idxs]
    y = conc[idxs]
    model = fitlog(x, y)
    λ = coef(model)[2]
  end
  if λ ≥ 0
    outlier = true
    @warn "The estimated slope is not negative, got $λ"
  end
  #-λ, points, maxrsq, outlier
  return -λ
end
