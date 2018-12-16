"""
    auclinear(C₁::Number, C₂::Number, t₁::Number, t₂::Number)

Compute area under the curve (AUC) in an interval by linear trapezoidal rule.
"""
@inline function auclinear(C₁::Number, C₂::Number, t₁::Number, t₂::Number)
  Δt = t₂ - t₁
  return ((C₁ + C₂) * Δt) / 2
end

"""
    auclog(C₁::Number, C₂::Number, t₁::Number, t₂::Number)

Compute area under the curve (AUC) in an interval by log-linear trapezoidal
rule.
"""
@inline function auclog(C₁::Number, C₂::Number, t₁::Number, t₂::Number)
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
    aumclinear(C₁::Number, C₂::Number, t₁::Number, t₂::Number)

Compute area under the first moment of the concentration (AUMC) in an interval
by linear trapezoidal rule.
"""
@inline function aumclinear(C₁::Number, C₂::Number, t₁::Number, t₂::Number)
  return auclinear(C₁*t₁, C₂*t₂, t₁, t₂)
end

"""
    aumclog(C₁::Number, C₂::Number, t₁::Number, t₂::Number)

Compute area under the first moment of the concentration (AUMC) in an interval
by log-linear trapezoidal rule.
"""
@inline function aumclog(C₁::Number, C₂::Number, t₁::Number, t₂::Number)
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

function _auc(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I}; kwargs...) where {C,T,AUC,AUMC,D<:AbstractArray,Z,F,N,I}
  dose = nca.dose
  map(eachindex(dose)) do i
    subj = subject_at_ithdose(nca, i)
    _auc(subj; kwargs...)
  end
end

function _auc(nca::NCASubject; interval=nothing, auctype, method=:linear, linear, log, inf, kwargs...)
  conc, time = nca.conc, nca.time
  isaucinf = auctype in (:AUMCinf, :AUCinf)
  !(method in (:linear, :linuplogdown, :linlog)) && throw(ArgumentError("method must be :linear, :linuplogdown or :linlog"))
  if !(interval === nothing)
    if isaucinf && isfinite(interval[2])
      #@warn "Requesting AUCinf when the end of the interval is not Inf"
      auctype = auctype === :AUCinf ? :AUClast : :AUMClast
      isaucinf = false
    end
    interval[1] >= interval[2] && throw(ArgumentError("The AUC interval must be increasing, got interval=$interval"))
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
    concstart = interpextrapconc(nca, lo; interpmethod=method, kwargs...)
    idx1, idx2 = let lo = lo, hi = hi
      findfirst(x->x>=lo, time),
      findlast( x->x<=hi, time)
    end
  else
    idx1, idx2 = firstindex(time), lastindex(time)
    lo, hi = first(time), last(time)
    concstart = conc[idx1]
  end
  # assert the type here because we know that `auc` won't be `missing`
  __auc = linear(concstart, conc[idx1], lo, time[idx1])
  auc::_auctype(nca, auctype) = __auc
  # auc of the bounary intervals
  if time[idx1] != lo # if the interpolation point does not hit the exact data point
    auc = intervalauc(concstart, conc[idx1], lo, time[idx1], idx1-1, nca.maxidx, method, linear, log)
    int_idxs = idx1:idx2-1
  else # we compute the first interval in the data
    auc = intervalauc(conc[idx1], conc[idx1+1], time[idx1], time[idx1+1], idx1, nca.maxidx, method, linear, log)
    int_idxs = idx1+1:idx2-1
  end
  if isfinite(hi)
    concend = interpextrapconc(nca, hi; interpmethod=method, kwargs...)
    if conc[idx2] != concend
      auc += intervalauc(conc[idx2], concend, time[idx2], hi, idx2, nca.maxidx, method, linear, log)
    end
  end

  _clast = clast(nca)
  _tlast = tlast(nca)
  if ismissing(_clast)
    if all(x->x<=nca.llq, conc)
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
  if isaucinf
    λz = lambdaz(nca; recompute=false, kwargs...)[1]
    aucinf′ = inf(_clast, _tlast, λz) # the extrapolation part
  else
    aucinf′= zero(auc)
  end

  # cache results
  if interval == nothing # only cache when interval is nothing
    if isaucinf
      if auctype === :AUCinf
        nca.auc_last = auc
        nca.auc_inf = auc + aucinf′
      else
        nca.aumc_last = auc
        nca.aumc_inf = auc + aucinf′
      end
    else
      if auctype === :AUClast
        nca.auc_last = auc
      else
        nca.aumc_last = auc
      end
    end
  end

  auc += aucinf′ # add the infinite back

  return auc
end

"""
  auc(nca::NCASubject; auctype::Symbol, method::Symbol, interval=nothing, kwargs...)

Compute area under the curve (AUC) by linear trapezoidal rule `(method =
:linear)` or by log-linear trapezoidal rule `(method = :linuplogdown)`. It
calculates AUC and normalized AUC if `dose` is provided.
"""
function auc(nca::NCASubject; auctype=:AUCinf, interval=nothing, kwargs...)
  if interval isa Tuple || interval === nothing
    return auc_nokwarg(nca, auctype, interval; kwargs...)
  elseif interval isa AbstractArray
    f = let nca=nca, auctype=auctype, kwargs=kwargs
      i->auc_nokwarg(nca, auctype, i; kwargs...)
    end
    return map(f, interval)
  else
    error("interval must be a tuple or an array of tuples")
  end
end

@inline function auc_nokwarg(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I}, auctype, interval; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I}
  dose = nca.dose
  sol = nothing
  auctype in (:AUCinf, :AUClast) || throw(ArgumentError("auctype must be either :AUCinf or :AUClast for auc"))
  if !(D <: AbstractArray)
    if auctype === :AUCinf && interval === nothing # if we can use the cached result
      nca.auc_inf === nothing || (sol = nca.auc_inf)
    elseif auctype === :AUClast
      !(nca.auc_last === nothing) && interval === nothing && (sol = nca.auc_last)
    end
  end
  if sol === nothing
    sol = _auc(nca; auctype=auctype, interval=interval, linear=auclinear, log=auclog, inf=aucinf, kwargs...)
  end
  return D === Nothing ? sol : (auc=sol, auc_dn=normalize.(sol, dose)) # if dose is not nothing, normalize AUC
end

"""
  aumc(nca::NCASubject; method::Symbol, interval=(0, Inf), kwargs...)

Compute area under the first moment of the concentration (AUMC) by linear
trapezoidal rule `(method = :linear)` or by log-linear trapezoidal rule
`(method = :linuplogdown)`. It calculates AUC and normalized AUC if `dose` is
provided.
"""
function aumc(nca; auctype=:AUMCinf, interval=nothing, kwargs...)
  if interval isa Tuple || interval === nothing
    return aumc_nokwarg(nca, auctype, interval; kwargs...)
  elseif interval isa AbstractArray
    f = let nca=nca, auctype=auctype, kwargs=kwargs
      i->aumc_nokwarg(nca, auctype, i; kwargs...)
    end
    return map(f, interval)
  else
    error("interval must be a tuple or an array of tuples")
  end
end

@inline function aumc_nokwarg(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I}, auctype,
                              interval; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I}
  dose = nca.dose
  sol = nothing
  if !(D <: AbstractArray)
    auctype in (:AUMCinf, :AUMClast) || throw(ArgumentError("auctype must be either :AUMCinf or :AUMClast for aumc"))
    if auctype === :AUMCinf && interval === nothing # if we can use the cached result
      nca.aumc_inf === nothing || (sol = nca.aumc_inf)
    elseif auctype === :AUMClast
      !(nca.aumc_last === nothing) && interval === nothing && (sol = nca.aumc_last)
    end
  end
  if sol === nothing
    sol = _auc(nca; auctype=auctype, interval=interval, linear=aumclinear, log=aumclog, inf=aumcinf, kwargs...)
  end
  return D === Nothing ? sol : (aumc=sol, aumc_dn=normalize.(sol, dose)) # if dose is not nothing, normalize AUMC
end

function auc_extrap_percent(nca::NCASubject; kwargs...)
  aucinf  = auc(nca; auctype=:AUCinf, kwargs...)[1]
  auclast = auc(nca; auctype=:AUClast, kwargs...)[1]
  @. (aucinf-auclast)/aucinf * 100
end

function aumc_extrap_percent(nca::NCASubject; kwargs...)
  aumcinf  = aumc(nca; auctype=:AUMCinf, kwargs...)[1]
  aumclast = aumc(nca; auctype=:AUMClast, kwargs...)[1]
  @. (aumcinf-aumclast)/aumcinf * 100
end

fitlog(x, y) = lm(hcat(fill!(similar(x), 1), x), log.(y[y.!=0]))

function lambdaz(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I}; kwargs...) where {C,T,AUC,AUMC,D<:AbstractArray,Z,F,N,I}
  obj = map(eachindex(nca.dose)) do i
    subj = subject_at_ithdose(nca, i)
    lambdaz(subj; kwargs...)
  end
  λ, intercept, points, r2 = (map(x->x[i], obj) for i in 1:4)
  (lambdaz=λ, intercept=intercept, points=points, r2=r2)
end

"""
  lambdaz(nca::NCASubject; threshold=10, idxs=nothing) -> (lambdaz, points, r2)

Calculate ``ΛZ``, ``r^2``, and the number of data points from the profile used
in the determination of ``ΛZ``.
"""
function lambdaz(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I};
                 threshold=10, idxs=nothing, slopetimes=nothing, recompute=true, kwargs...
                )::NamedTuple{(:lambdaz, :intercept, :points, :r2),Tuple{Z,F,Int,F}} where {C,T,AUC,AUMC,D,Z,F,N,I}
  if !(nca.lambdaz === nothing) && !recompute
    return (lambdaz=nca.lambdaz, intercept=nca.intercept, points=nca.points, r2=nca.r2)
  end
  !(idxs === nothing) && !(slopetimes === nothing) && throw(ArgumentError("you must provide only one of idxs or slopetimes"))
  conc, time = nca.conc, nca.time
  maxr2::F = oneunit(F)*false
  λ::Z = oneunit(Z)*false
  NoUnitEltype = typeof(one(Z))
  time′ = reinterpret(NoUnitEltype, time)
  conc′ = reinterpret(NoUnitEltype, conc)
  points = 2
  outlier = false
  _, cmaxidx = conc_maximum(conc, eachindex(conc))
  n = length(time)
  if slopetimes === nothing && idxs === nothing
    m = min(n-1, threshold, n-cmaxidx-1)
    m < 2 && throw(ArgumentError("lambdaz must be calculated from at least three data points after Cmax"))
    for i in 2:m
      x = time′[end-i:end]
      y = conc′[end-i:end]
      model = fitlog(x, y)
      r2 = oneunit(F) * r²(model)
      if r2 > maxr2
        maxr2 = r2
        λ = oneunit(Z) * coef(model)[2]
        intercept = coef(model)[1]
        points = i+1
      end
    end
  else
    points = idxs === nothing ? length(slopetimes) : length(idxs)
    points < 3 && throw(ArgumentError("lambdaz must be calculated from at least three data points"))
    idxs === nothing && (idxs = findall(x->x in slopetimes, time))
    length(idxs) != points && throw(ArgumentError("elements slopetimes must occur in nca.time"))
    x = time′[idxs]
    y = conc′[idxs]
    model = fitlog(x, y)
    λ = oneunit(Z) * coef(model)[2]
    intercept = coef(model)[1]
    r2 = oneunit(F) * r²(model)
  end
  if λ ≥ zero(λ)
    outlier = true
    @warn "The estimated slope is not negative, got $λ"
  end
  nca.lambdaz=-λ
  nca.points=points
  nca.r2=maxr2
  nca.intercept=intercept
  return (lambdaz=-λ, intercept=intercept, points=points, r2=maxr2)
end
