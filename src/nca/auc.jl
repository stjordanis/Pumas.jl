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
@noinline function auclog(C₁::Number, C₂::Number, t₁::Number, t₂::Number)
  Δt = t₂ - t₁
  ΔC = C₂ - C₁
  lg = log(C₂/C₁)
  return (Δt * ΔC) / lg
end


"""
    extrapaucinf(clast, tlast, lambdaz)
idx = 1:16

Extrapolate auc to the infinite.
"""
@inline extrapaucinf(clast, tlast, lambdaz) = clast/lambdaz

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
@noinline function aumclog(C₁::Number, C₂::Number, t₁::Number, t₂::Number)
  Δt = t₂ - t₁
  lg = log(C₂/C₁)
  return Δt*(t₂*C₂ - t₁*C₁)/lg - Δt^2*(C₂-C₁)/lg^2
end

"""
    extrapaumcinf(clast, tlast, lambdaz)

Extrapolate the first moment to the infinite.
"""
@inline extrapaumcinf(clast, tlast, lambdaz) = clast*tlast/lambdaz + clast/lambdaz^2

@enum AUCmethod Linear Log

@inline function choosescheme(c1, c2, t1, t2, i::Int, maxidx::Int, method::Symbol)
  if method === :linear ||
    (method === :linuplogdown && c2>=c1) && (method === :linlog && maxidx)
    return Linear
  else
    c1 <= zero(c1) || c2 <= zero(c2) && return Linear
    return Log
  end
end

@inline function intervalauc(c1, c2, t1, t2, i::Int, maxidx::Int, method::Symbol, linear, log)
  m = choosescheme(c1, c2, t1, t2, i, maxidx, method)
  return m === Linear ? linear(c1, c2, t1, t2) : log(c1, c2, t1, t2)
end

@inline function iscached(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}, sym::Symbol) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}
  @inbounds begin
    # `points` is initialized to 0
    sym === :lambdaz && return !(nca.points[1] === 0)
    # `auc_last` and `aumc_last` are initialized to -1
    sym === :auc     && return !(nca.auc_last[1]  === -oneunit(eltype(AUC)))
    sym === :aumc    && return !(nca.aumc_last[1] === -oneunit(eltype(AUMC)))
  end
end

function _auc(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}, interval, linear, log, inf, ret_typ;
              auctype, method=:linear, isauc, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}
  # fast return
  if interval === nothing
    if auctype === :inf
      _clast = clast(nca; kwargs...)
      _tlast = tlast(nca)
      aucinf′ = inf(_clast, _tlast, lambdaz(nca; recompute=false, kwargs...))
      isauc  && iscached(nca, :auc)  && return (nca.auc_last[1] + aucinf′) ::ret_typ
      !isauc && iscached(nca, :aumc) && return (nca.aumc_last[1] + aucinf′)::ret_typ
    elseif auctype === :last
      isauc  && iscached(nca, :auc)  && return nca.auc_last[1] ::ret_typ
      !isauc && iscached(nca, :aumc) && return nca.aumc_last[1]::ret_typ
    end
  end
  conc, time = nca.conc, nca.time
  auctype in (:inf, :last) || throw(ArgumentError("auctype must be either :inf or :last"))
  isinf = auctype === :inf
  !(method in (:linear, :linuplogdown, :linlog)) && throw(ArgumentError("method must be :linear, :linuplogdown or :linlog"))
  if !(interval === nothing)
    if isinf && isfinite(interval[2])
      auctype === :inf && (auctype = :last)
      isinf = false
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
    concstart = interpextrapconc(nca, lo; method=method, kwargs...)
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
  __auc = linear(concstart, conc[idx1], lo, time[idx1]) # this must be zero
  auc::ret_typ = __auc
  # auc of the bounary intervals
  if time[idx1] != lo # if the interpolation point does not hit the exact data point
    auc = intervalauc(concstart, conc[idx1], lo, time[idx1], idx1-1, nca.maxidx, method, linear, log)
    int_idxs = idx1:idx2-1
  elseif !isassigned(time, idx1+1) # if idx1+1 is not assigned
    int_idxs = 0:-1 # no iteration
  else # we compute the first interval in the data
    auc = intervalauc(conc[idx1], conc[idx1+1], time[idx1], time[idx1+1], idx1, nca.maxidx, method, linear, log)
    int_idxs = idx1+1:idx2-1
  end
  if isfinite(hi)
    concend = interpextrapconc(nca, hi; method=method, kwargs...)
    if conc[idx2] != concend
      auc += intervalauc(conc[idx2], concend, time[idx2], hi, idx2, nca.maxidx, method, linear, log)
    end
  end

  _clast = clast(nca; kwargs...)
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
  if isinf
    λz = lambdaz(nca; recompute=false, kwargs...)
    aucinf′ = inf(_clast, _tlast, λz) # the extrapolation part
  else
    aucinf′= zero(auc)
  end

  # cache results
  if interval == nothing # only cache when interval is nothing
    if isauc
      AUC <: AbstractArray  ? nca.auc_last[1] = auc  : nca.auc_last = auc
    else
      AUMC <: AbstractArray ? nca.aumc_last[1] = auc : nca.aumc_last = auc
    end
  end

  auc += aucinf′ # add the infinite part

  return auc
end

"""
  auc(nca::NCASubject; auctype::Symbol, method::Symbol, interval=nothing, kwargs...)

Compute area under the curve (AUC) by linear trapezoidal rule `(method =
:linear)` or by log-linear trapezoidal rule `(method = :linuplogdown)`.
"""
function auc(nca::NCASubject; auctype=:inf, interval=nothing, kwargs...)
  if interval isa Tuple || interval === nothing
    return auc_nokwarg(nca, auctype, interval; kwargs...)
  elseif interval isa AbstractArray
    f = let nca=nca, auctype=auctype, kwargs=kwargs
      i->auc_nokwarg(nca, auctype, i; kwargs...)
    end
    return map(f, interval)
  else
    error("""
          interval must be a tuple or an array of tuples, e.g.,
          auc(nca; interval=[(0,1.), (2, 3.)])
          """)
  end
end

@inline function auc_nokwarg(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}, auctype, interval; kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}
  sol = _auc(nca, interval, auclinear, auclog, extrapaucinf, eltype(AUC); auctype=auctype, isauc=true, kwargs...)
  return sol
end

"""
  aumc(nca::NCASubject; method::Symbol, interval=(0, Inf), kwargs...)

Compute area under the first moment of the concentration (AUMC) by linear
trapezoidal rule `(method = :linear)` or by log-linear trapezoidal rule
`(method = :linuplogdown)`.
"""
function aumc(nca; auctype=:inf, interval=nothing, kwargs...)
  if interval isa Tuple || interval === nothing
    return aumc_nokwarg(nca, auctype, interval; kwargs...)
  elseif interval isa AbstractArray
    f = let nca=nca, auctype=auctype, kwargs=kwargs
      i->aumc_nokwarg(nca, auctype, i; kwargs...)
    end
    return map(f, interval)
  else
    error("""
          interval must be a tuple or an array of tuples, e.g.,
          aumc(nca; interval=[(0,1.), (2, 3.)])
          """)
  end
end

@inline function aumc_nokwarg(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}, auctype, interval; kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}
  sol = _auc(nca, interval, aumclinear, aumclog, extrapaumcinf, eltype(AUMC); auctype=auctype, isauc=false, kwargs...)
  return sol
end

function auc_extrap_percent(nca::NCASubject; kwargs...)
  aucinf  = auc(nca; auctype=:inf, kwargs...)
  auclast = auc(nca; auctype=:last, kwargs...)
  @. (aucinf-auclast)/aucinf * 100
end

function aumc_extrap_percent(nca::NCASubject; kwargs...)
  aumcinf  = aumc(nca; auctype=:inf, kwargs...)
  aumclast = aumc(nca; auctype=:last, kwargs...)
  @. (aumcinf-aumclast)/aumcinf * 100
end

fitlog(x, y) = lm(hcat(fill!(similar(x), 1), x), log.(y[y.!=0]))

"""
  lambdaz(nca::NCASubject; threshold=10, idxs=nothing) -> lambdaz

Calculate terminal elimination rate constant ``λz``.
"""
function lambdaz(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID};
                 threshold=10, idxs=nothing, slopetimes=nothing, recompute=true, kwargs...
                )::eltype(Z) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}
  if iscached(nca, :lambdaz) && !recompute
    return lambdaz=first(nca.lambdaz)
  end
  _F = eltype(F)
  _Z = eltype(Z)
  !(idxs === nothing) && !(slopetimes === nothing) && throw(ArgumentError("you must provide only one of idxs or slopetimes"))
  conc, time = nca.conc, nca.time
  maxadjr2::_F = -oneunit(_F)*true
  λ::_Z = oneunit(_Z)*false
  time′ = reinterpret(typeof(one(eltype(time))), time)
  conc′ = reinterpret(typeof(one(eltype(conc))), conc)
  outlier = false
  _, cmaxidx = conc_maximum(conc, eachindex(conc))
  n = length(time)
  if slopetimes === nothing && idxs === nothing
    m = min(n-1, threshold, n-cmaxidx-1)
    m < 2 && throw(ArgumentError("ID $(nca.id) errored: lambdaz must be calculated from at least three data points after Cmax"))
    idx2 = length(time′)
    for i in 2:m
      idx1 = idx2-i
      idxs = idx1:idx2
      x = time′[idxs]
      y = conc′[idxs]
      model = fitlog(x, y)
      adjr2 = oneunit(_F) * adjr²(model)
      if adjr2 > maxadjr2
        maxadjr2 = adjr2
        r2 = oneunit(_F) * r²(model)
        λ = oneunit(_Z) * coef(model)[2]
        intercept = coef(model)[1]
        firstpoint = time[idxs[1]]
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
    λ = oneunit(_Z) * coef(model)[2]
    intercept = coef(model)[1]
    r2 = oneunit(_F) * r²(model)
    firstpoint = time[idxs[1]]
    maxadjr2 = oneunit(_F) * adjr²(model)
  end
  if λ ≥ zero(λ)
    outlier = true
    @warn "The estimated slope is not negative, got $λ"
  end
  if Z <: AbstractArray
    nca.lambdaz[1] = -λ
    nca.points[1] = points
    nca.firstpoint[1] = firstpoint
    nca.r2[1] = r2
    nca.adjr2[1] = maxadjr2
    nca.intercept[1] = intercept
  else
    nca.lambdaz = -λ
    nca.points = points
    nca.firstpoint = firstpoint
    nca.r2 = r2
    nca.adjr2 = maxadjr2
    nca.intercept = intercept
  end
  return lambdaz=-λ
end

"""
  lambdaznpoints(nca::NCASubject; kwargs...)

Give the number of points that is used in the `λz` calculation. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdaznpoints(nca::NCASubject; kwargs...) = (lambdaz(nca; kwargs...); first(nca.points))

"""
  lambdaztimefirst(nca::NCASubject; kwargs...)

Give the first time point that is used in the `λz` calculation. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdaztimefirst(nca::NCASubject; kwargs...) = (lambdaz(nca; kwargs...); first(nca.firstpoint))

"""
  lambdazintercept(nca::NCASubject; kwargs...)

Give the `y`-intercept in the log-linear scale when calculating `λz`. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdazintercept(nca::NCASubject; kwargs...) = (lambdaz(nca; kwargs...); first(nca.intercept))


"""
  lambdazr2(nca::NCASubject; kwargs...)

Give the coefficient of determination (``r²``) when calculating `λz`. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdazr2(nca::NCASubject; kwargs...) = (lambdaz(nca; kwargs...); first(nca.r2))

"""
  lambdazadjr2(nca::NCASubject; kwargs...)

Give the adjusted coefficient of determination (``adjr²``) when calculating
`λz`. Please calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdazadjr2(nca::NCASubject; kwargs...) = (lambdaz(nca; kwargs...); first(nca.adjr2))
