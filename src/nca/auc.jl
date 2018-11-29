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
rule.
"""
@inline function auclog(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
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
by log-linear trapezoidal rule.
"""
@inline function aumclog(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
  lg = log(C₂/C₁)
  return Δt*(t₂*C₂ - t₁*C₁)/lg - Δt^2*(C₂-C₁)/lg^2
end

"""
    aumcinf(clast, tlast, lambdaz)

Extrapolate the first moment to the infinite.
"""
@inline aumcinf(clast, tlast, lambdaz) = clast*tlast/lambdaz + clast/lambdaz^2

@enum AUCmethod Linear Log

function choosescheme(c1, c2, t1, t2, i::Int, maxidx::Int, method::Symbol)
  if method === :linear ||
    (method === :linuplogdown && c2>=c1) && (method === :linlog && maxidx)
    return Linear
  else
    c1 <= zero(c1) || c2 <= zero(c2) && return Linear
    return Log
  end
end

function intervalauc(c1, c2, t1, t2, i::Int, maxidx::Int, method::Symbol, linear, log)
  m = choosescheme(c1, c2, t1, t2, i, maxidx, method)
  return m === Linear ? linear(c1, c2, t1, t2) : log(c1, c2, t1, t2)
end

@inline function _auc(nca::NCAdata; interval=nothing, auctype::Symbol, method::Symbol=:linear, linear, log, inf, kwargs...)
  conc, time = nca.conc, nca.time
  !(method in (:linear, :linuplogdown, :linlog)) && throw(ArgumentError("method must be :linear, :linuplogdown or :linlog"))
  if !(interval === nothing)
    interval[1] >= interval[2] && throw(ArgumentError("The AUC interval must be increasing, got interval=$interval"))
    #auctype in (:AUCinf, :AUMCinf) && isfinite(interval[2]) && @warn "Requesting AUCinf when the end of the interval is not Inf"
    lo, hi = interval
    lo < first(time) && @warn "Requesting an AUC range starting $lo before the first measurement $(first(time)) is not allowed"
    lo > last(time) && @warn "AUC start time $lo is after the maximum observed time $(last(time))"
    # TODO: handle `auclinlog` with IV bolus.
    #
    # `C0` is the concentration at time zero if the route of administration is IV
    # (intranvenous). If concentration at time zero is not measured after the IV
    # bolus, a linear back-extrapolation is done to get the intercept which
    # represents concentration at time zero (`C0`)
    #
    concstart = interpextrapconc(nca, lo, interpmethod=method, extrapmethod=:AUCinf)
    idx1, idx2 = let lo = lo, hi = hi
      findfirst(x->x>=lo, time),
      findlast( x->x<=hi, time)
    end
  else
    idx1, idx2 = firstindex(time), lastindex(time)
    lo, hi = first(time), last(time)
    concstart = conc[idx1]
  end
  # auc of the bounary intervals
  __auc = linear(concstart, conc[idx1], lo, time[idx1])
  auc::_auctype(nca, auctype) = __auc
  if conc[idx1] != concstart
    auc = intervalauc(concstart, conc[idx1], lo, time[idx1], idx1-1, nca.maxidx, method, linear, log)
    int_idxs = idx1+1:idx2-1
  else
    auc = intervalauc(conc[idx1], conc[idx1+1], time[idx1], time[idx1+1], idx1, nca.maxidx, method, linear, log)
    int_idxs = idx1+1:idx2-1
  end
  if isfinite(hi)
    concend = interpextrapconc(nca, hi, interpmethod=method, extrapmethod=:AUCinf)
    if conc[idx2] != concend
      auc += intervalauc(conc[idx2], concend, time[idx2], hi, idx2, nca.maxidx, method, linear, log)
    end
  end

  _clast = clast(nca)
  _tlast = tlast(nca)
  if ismissing(_clast)
    if all(x<-x<=nca.llq, conc)
      return zero(auc)
    else
      error("Unknown error with missing `tlast` but non-BLQ concentrations")
    end
  end

  # Compute the AUxC
  # Include the first point after `tlast` if it exists and we are computing AUCall
  # do not support :AUCall for now
  # auctype === :AUCall && tlast > hi && (idx2 = idx2 + 1)
  for i in int_idxs
    auc += intervalauc(conc[i], conc[i+1], time[i], time[i+1], i, nca.maxidx, method, linear, log)
  end

  # cache results
  if auctype === :AUMCinf || auctype === :AUCinf
    λz = lambdaz(nca; kwargs...)[1]
    aucinf′ = inf(_clast, _tlast, λz)
    if auctype === :AUCinf
      nca.auc_last = auc
      auc += aucinf′
      nca.auc_inf = auc
    else
      nca.aumc_last = auc
      auc += aucinf′
      nca.aumc_inf = auc
    end
  elseif interval === nothing
    if auctype === :AUClast
      nca.auc_last = auc
    else
      nca.aumc_last = auc
    end
  end
  return auc
end

"""
  auc(nca::NCAdata; auctype::Symbol, method::Symbol, interval=nothing, kwargs...)

Compute area under the curve (AUC) by linear trapezoidal rule `(method =
:linear)` or by log-linear trapezoidal rule `(method = :linuplogdown)`. It
calculates AUC and normalized AUC if `does` is provided.
"""
function auc(nca::NCAdata; auctype=:AUCinf, interval=nothing, kwargs...)
  dose = nca.dose
  # fast return
  sol = nothing
  if auctype === :AUCinf
    nca.auc_inf === nothing || (sol = nca.auc_inf)
  elseif auctype === :AUClast
    !(nca.auc_last === nothing) && interval === nothing && (sol = nca.auc_last)
  else
    throw(ArgumentError("auctype must be either :AUCinf or :AUClast"))
  end
  if sol === nothing
    sol = _auc(nca; auctype=auctype, interval=interval, linear=auclinear, log=auclog, inf=aucinf, kwargs...)
  end
  return dose === nothing ? sol : (sol, sol/dose)
end

"""
  aumc(nca::NCAdata; method::Symbol, interval=(0, Inf) kwargs...)

Compute area under the first moment of the concentration (AUMC) by linear
trapezoidal rule `(method = :linear)` or by log-linear trapezoidal rule
`(method = :linuplogdown)`. It calculates AUC and normalized AUC if `does` is
provided.
"""
function aumc(nca::NCAdata; auctype=:AUMCinf, interval=nothing, kwargs...)
  dose = nca.dose
  # fast return
  sol = nothing
  if auctype === :AUMCinf
    nca.aumc_inf === nothing || (sol = nca.aumc_inf)
  elseif auctype === :AUMClast
    !(nca.aumc_last === nothing) && interval === nothing && (sol = nca.aumc_last)
  else
    throw(ArgumentError("auctype must be either :AUMCinf or :AUMClast"))
  end
  if sol === nothing
    sol = _auc(nca; auctype=auctype, interval=interval, linear=aumclinear, log=aumclog, inf=aumcinf, kwargs...)
  end

  return dose === nothing ? sol : (sol, sol/dose)
end

function auc_extrap_percent(nca::NCAdata; kwargs...)
  auc(nca; auctype=:AUCinf, kwargs...)
  (nca.auc_inf-nca.auc_last)/nca.auc_inf * 100
end

function aumc_extrap_percent(nca::NCAdata; kwargs...)
  aumc(nca; auctype=:AUMCinf, kwargs...)
  (nca.aumc_inf-nca.aumc_last)/nca.aumc_inf * 100
end

fitlog(x, y) = lm(hcat(fill!(similar(x), 1), x), log.(y[y.!=0]))

"""
  lambdaz(conc, time; threshold=10, check=true, idxs=nothing,
  missingconc=:drop) -> (lambdaz, points, r2)

Calculate ``λz``, ``r^2``, and the number of data points from the profile used
in the determination of ``λz``.
"""
function lambdaz(nca::NCAdata; threshold=10, idxs=nothing, kwargs...)
  !(nca.lambdaz === nothing) && return (nca.lambdaz, nca.points, nca.r2)
  conc, time = nca.conc, nca.time
  maxr2 = 0.0
  lambdaz = 0.0
  points = 2
  outlier = false
  _, cmaxidx = conc_maximum(conc, eachindex(conc))
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
  nca.lambdaz=-λ
  nca.points=points
  nca.r2=r2
  return (lambdaz=nca.lambdaz, points=nca.points, r2=nca.r2)
end
