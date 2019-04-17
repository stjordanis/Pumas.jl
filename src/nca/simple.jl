"""
  clast(nca::NCASubject)

Calculate `clast`
"""
function clast(nca::NCASubject; pred=false, kwargs...)
  if pred
    λz = lambdaz(nca; recompute=false, kwargs...)
    intercept = lambdazintercept(nca; recompute=false, kwargs...)
    return exp(intercept - λz*tlast(nca))*oneunit(eltype(nca.conc))
  else
    idx = nca.lastidx
    return idx === -1 ? missing : nca.conc[idx]
  end
end

"""
  tlast(nca::NCASubject)

Calculate `tlast`
"""
function tlast(nca::NCASubject; kwargs...)
  idx = nca.lastidx
  return idx === -1 ? missing : nca.time[idx]
end

# This function uses ``-1`` to denote missing as after checking `conc` is
# strictly great than ``0``.
function ctlast_idx(conc, time; llq=nothing, check=true)
  check && checkconctime(conc, time)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findlast(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

function ctfirst_idx(conc, time; llq=nothing, check=true)
  check && checkconctime(conc, time)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findfirst(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

"""
  tmax(nca::NCASubject; interval=(0.,Inf), kwargs...)

Calculate ``T_{max}_{t_1}^{t_2}``
"""
function tmax(nca::NCASubject; interval=(0.,Inf), kwargs...)
  if interval isa Tuple
    return ctmax(nca; interval=interval, kwargs...)[2]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctmax(nca; interval=i, kwargs...)[2]
    end
    map(f, interval)
  end
end

"""
  cmax(nca::NCASubject; interval=(0.,Inf), kwargs...)

Calculate ``C_{max}_{t_1}^{t_2}``
"""
function cmax(nca::NCASubject; interval=(0.,Inf), kwargs...)
  dose = nca.dose
  if interval isa Tuple
    sol = ctmax(nca; interval=interval, kwargs...)[1]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctmax(nca; interval=i, kwargs...)[1]
    end
    sol = map(f, interval)
  end
  sol
end

"""
  cmin(nca::NCASubject; kwargs...)

Calculate minimum observed concentration
"""
function cmin(nca::NCASubject; kwargs...) # TODO: clowest C(tau)
  conc, time = nca.conc, nca.time
  val, _ = conc_maximum(conc, eachindex(conc), true)
  return val
end

"""
  tmin(nca::NCASubject; kwargs...)

Calculate time of minimum observed concentration
"""
function tmin(nca::NCASubject; kwargs...)
  conc, time = nca.conc, nca.time
  _, idx = conc_maximum(conc, eachindex(conc), true)
  return time[idx]
end

@inline function ctmax(nca::NCASubject; interval=(0.,Inf), kwargs...)
  conc, time = nca.conc, nca.time
  if interval === (0., Inf)
    val, idx = conc_maximum(conc, eachindex(conc))
    return (cmax=val, tmax=time[idx])
  end
  @assert interval[1] < interval[2] "t0 must be less than t1"
  interval[1] > time[end] && throw(ArgumentError("t0 is longer than observation time"))
  idx1, idx2 = let (lo, hi)=interval
    findfirst(t->t>=lo, time),
    findlast( t->t<=hi, time)
  end
  val, idx = conc_maximum(conc, idx1:idx2)
  return (cmax=val, tmax=time[idx])
end

@inline function conc_maximum(conc, idxs, min::Bool=false)
  idx = 1
  val = -oneunit(Base.nonmissingtype(eltype(conc)))
  min && (val *= -Inf)
  for i in idxs
    if !ismissing(conc[i])
      if min
        val > conc[i] && (val = conc[i]; idx=i)
      else
        val < conc[i] && (val = conc[i]; idx=i)
      end
    end
  end
  return val, idx
end

"""
  thalf(nca::NCASubject; kwargs...)

Calculate half life time.
"""
thalf(nca::NCASubject; kwargs...) = log(2)./lambdaz(nca; recompute=false, kwargs...)

"""
  clf(nca::NCASubject; kwargs...)

Calculate total drug clearance divided by the bioavailability (F), which is just the
inverse of the dose normalizedosed ``AUC_0^∞`` or ``AUC_0^τ`` (steady-state).
"""
function clf(nca::NCASubject; ss=false, kwargs...)
  nca.dose === nothing && return missing
  if ss
    map(inv, normalizedose(auctau(nca; kwargs...), nca))
  else
    map(inv, normalizedose(auc(nca; kwargs...), nca))
  end
end

"""
  vss(nca::NCASubject; kwargs...)

Calculate apparent volume of distribution at equilibrium for IV bolus doses.
``V_{ss} = AUMC / {AUC_0^\\inf}^2`` for dose normalizedosed `AUMC` and `AUC`.
"""
function vss(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  normalizedose(aumc(nca; kwargs...), nca) ./ (normalizedose(auc(nca; kwargs...), nca)).^2
end

"""
  vz(nca::NCASubject; kwargs...)

Calculate the volume of distribution during the terminal phase.
``V_z = 1/(AUC_0^τ⋅λz)`` for dose normalizedosed `AUC`.
"""
function vz(nca::NCASubject; ss=false, kwargs...)
  nca.dose === nothing && return missing
  λ = lambdaz(nca; recompute=false, kwargs...)
  if ss
    τauc = auctau(nca; kwargs...)
    @. inv(normalizedose(τauc, nca.dose) * λ)
  else
    aucinf = normalizedose(auc(nca; kwargs...), nca)
    @. inv(aucinf * λ)
  end
end

"""
  bioav(nca::NCASubject; ithdose::Integer, kwargs...)

Bioavailability is the ratio of two AUC values.
``Bioavailability (F) = (AUC_0^\\infty_{po}/Dose_{po})/(AUC_0^\\infty_{iv}/Dose_{iv})``
"""
function bioav(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,II}; ithdose=missing, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,II}
  ismissing(ithdose) && return missing
  multidose = nca.dose isa AbstractArray
  # if there is only a single dose
  multidose || return missing
  # if we only have IV or EV
  length(unique(getfield.(nca.dose, :formulation))) == 1 && return fill(missing, length(nca.dose))
  # initialize
  auc_0_inf_po = auc_0_inf_iv = zero(eltype(AUC))/oneunit(first(nca.dose).amt) # normalizedosed
  sol = zeros(typeof(ustrip(auc_0_inf_po)), axes(nca.dose))
  refdose = subject_at_ithdose(nca, ithdose)
  refauc  = normalizedose(auc(refdose; auctype=:inf, kwargs...), refdose)
  map(eachindex(nca.dose)) do idx
    subj = subject_at_ithdose(nca, idx)
    sol[idx] = normalizedose(auc(subj; auctype=:inf, kwargs...), subj)/refauc
  end
  return sol
end

#TODO: check the units
"""
  cl(nca::NCASubject; ithdose::Integer, kwargs...)

Total drug clearance
"""
function cl(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,II}; ss=false, ithdose=missing, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,II}
  nca.dose === nothing && return missing
  _clf = clf(nca; ss=ss, kwargs...)
  dose = nca.dose
  if nca.dose isa NCADose # single dose
    isiv(dose.formulation) || return missing
    _bioav = one(AUC)
    return _bioav*_clf
  else # multiple doses
    # TODO: check back on logic for this ithdose for CL, not sure about the user specification
    ismissing(ithdose) && throw(ArgumentError("`ithdose` must be provided for computing CL"))
    map(eachindex(dose)) do idx
      subj = subject_at_ithdose(nca, idx)
      formulation = subj.dose.formulation
      if idx == ithdose
        isiv(formulation) || throw(ArgumentError("the formulation of `ithdose` must be IV"))
      end
      if isiv(formulation)
        one(eltype(AUC))*_clf[idx]
      else
        missing
      end
    end # end multidoses
  end # end if
end

"""
  tlag(nca::NCASubject; kwargs...)

The time prior to the first increase in concentration.
"""
function tlag(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  isiv(nca.dose.formulation) && return missing
  idx = max(1, findfirst(c->c > nca.llq, nca.conc)-1)
  return nca.time[idx]
end

"""
  mrt(nca::NCASubject; kwargs...)

Mean residence time from the time of dosing to the time of the last measurable
concentration.

IV infusion:
  ``AUMC/AUC - TI/2`` where ``TI`` is the length of infusion.
non-infusion:
  ``AUMC/AUC``
"""
function mrt(nca::NCASubject; ss=false, kwargs...)
  dose = nca.dose
  dose === nothing && return missing
  ti2 = dose.duration*1//2
  if ss
    τ = tau(nca; kwargs...)
    aumcτ = aumctau(nca; kwargs...)
    aucτ = aumctau(nca; kwargs...)
    aumcinf = aumctau(nca; auctype=:inf, kwargs...)
    quotient = (aumcτ + τ*(aucinf - aucτ)) / aucτ
    dose.formulation === IVInfusion && (quotient -= ti2)
    return quotient
  else
    quotient = aumc(nca; kwargs...) / auc(nca; kwargs...)
    dose.formulation === IVInfusion && (quotient -= ti2)
    return quotient
  end
end

"""
  mat(nca::NCASubject; kwargs...)

Mean absorption time:
``MAT = MRT_po - MRT_iv``

For multiple dosing only.
"""
function mat(nca::NCASubject; kwargs...)
  # dose is checked in `mrt`, so we don't need to check it in `mat`
  multidose = nca.dose isa AbstractArray
  #multidose || error("Need more than one type of dose to calculate MAT")
  multidose || return missing
  mrt_po = mrt_iv = zero(eltype(eltype(nca.time)))
  for idx in eachindex(nca.dose)
    subj = subject_at_ithdose(nca, idx)
    if isiv(subj.dose.formulation)
      mrt_iv += mrt(subj; kwargs...)
    else
      mrt_po += mrt(subj; kwargs...)
    end
  end # end for
  mrt_po - mrt_iv
end

"""
  tau(nca::NCASubject)

Dosing interval. For multiple dosing only.
"""
function tau(nca::NCASubject; kwargs...) # warn if not provided
  has_ii(nca) && return nca.ii
  dose = nca.dose
  dose === nothing && return missing
  dose isa NCADose && return nca.abstime[nca.lastidx]-dose.time
  return dose[end].time-dose[end-1].time
end

"""
  cavg(nca::NCASubject)

Average concentration over one period. ``C_{avg} = AUC_{tau}/Tau``. For multiple dosing only.
"""
function cavg(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  subj = nca.dose isa NCADose ? nca : subject_at_ithdose(nca, 1)
  return auctau(subj; kwargs...)/tau(subj; kwargs...)
end

"""
  fluctation(nca::NCASubject; kwargs...)

Peak trough fluctuation over one dosing interval at steady state.
``Fluctuation = 100*(C_{max} - C_{min})/C_{avg}``
"""
fluctation(nca::NCASubject; kwargs...) = 100*(cmax(nca) - cmin(nca))/cavg(nca; kwargs...)

"""
  accumulationindex(nca::NCASubject; kwargs...)

Theoretical accumulation ratio. ``Accumulation_index = 1/(1-exp(-λ_z*Tau))``.
"""
function accumulationindex(nca::NCASubject; kwargs...)
  tmp = -lambdaz(nca; recompute=false, kwargs...)*tau(nca)
  one(tmp)/(one(tmp)-exp(tmp))
end

"""
  swing(nca::NCASubject; kwargs...)

  ``swing = (C_{max}-C_{min})/C_{min}``
"""
function swing(nca::NCASubject; kwargs...)
  _cmin = cmin(nca)
  sw = (cmax(nca) - _cmin) ./ _cmin
  isinf(sw) ? missing : sw
end

"""
  c0(nca::NCASubject; c0method=(:c0, :logslope, :c1, :cmin, :set0), kwargs...)

Estimate the concentration at dosing time for an IV bolus dose. It takes the
following methods:

- c0: If the observed `conc` at `dose.time` is nonzero, return that. This method should usually be used first for single-dose IV bolus data in case nominal time zero is measured.
- logslope: Compute the semilog line between the first two measured times, and use that line to extrapolate backward to `dose.time`.
- c1: Use the first point after `dose.time`.
- cmin: Set c0 to cmin during the interval. This method should usually be used for multiple-dose oral data and IV infusion data.
- set0: Set c0 to zero (regardless of any other data). This method should usually be used first for single-dose oral data.
"""
function c0(nca::NCASubject; c0method=(:c0, :logslope, :c1, :cmin, :set0), kwargs...)
  ret = missing
  # if we get one method, then convert it to a tuple anyway
  c0method isa Symbol && (c0method = tuple(c0method))
  dosetime = nca.dose === nothing ? zero(eltype(nca.time)) : first(nca.dose).time
  nca = nca.dose isa AbstractArray ? subject_at_ithdose(nca, 1) : nca
  #nca.dose !== nothing && nca.dose.formulation === EV && return missing
  while ismissing(ret) && !isempty(c0method)
    current_method = c0method[1]
    c0method = Base.tail(c0method)
    if current_method == :c0
      ret = _c0_method_c0(nca, dosetime)
    elseif current_method == :logslope
      ret = _c0_method_logslope(nca, dosetime)
    elseif current_method == :c1
      ret = _c0_method_c1(nca, dosetime)
    elseif current_method == :cmin
      ret = cmin(nca, dosetime)
    elseif current_method == :set0
      ret = nca.dose isa AbstractArray ? fill(zero(eltype(eltype(nca.conc))), length(nca.dose)) : zero(eltype(nca.conc))
    else
      throw(ArgumentError("unknown method $current_method, please use any combination of :c0, :logslope, :c1, :cmin, :set0. E.g. `c0(subj, c0method=(:c0, :cmin, :set0))`"))
    end
  end
  return ret
end

function _c0_method_c0(nca::NCASubject, dosetime)
  for i in eachindex(nca.time)
    c = nca.conc[i]
    t = nca.time[i]
    t == dosetime && !ismissing(c) && !iszero(c) && return c
  end
  return missing
end

function _c0_method_logslope(nca::NCASubject, dosetime)
  idxs = findall(eachindex(nca.time)) do i
    nca.time[i] > dosetime && !ismissing(nca.conc[i])
  end
  length(idxs) < 2 && return missing
  # If there is enough data, proceed to calculate
  c1 = ustrip(nca.conc[idxs[1]]); t1 = ustrip(nca.time[idxs[1]])
  c2 = ustrip(nca.conc[idxs[2]]); t2 = ustrip(nca.time[idxs[2]])
  if c2 < c1 && !iszero(c1)
    return exp(log(c1) - (t1 - ustrip(dosetime))*(log(c2)-log(c1))/(t2-t1))*oneunit(eltype(nca.conc))
  else
    return missing
  end
end

function _c0_method_c1(nca::NCASubject, dosetime)
  idx = findfirst(eachindex(nca.time)) do i
    nca.time[i] > dosetime && !ismissing(nca.conc[i])
  end
  return idx isa Number ? nca.conc[idx] : missing
end

# TODO: user input lambdaz, clast, and tlast?
# TODO: multidose?
function superposition(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,II},
                       tau::Number, ntau=Inf, args...;
                       doseamount=nothing, additionaltime::Vector=T[],
                       steadystatetol::Number=1e-3, method=:linear,
                       kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,II}
  D === Nothing && throw(ArgumentError("Dose must be known to compute superposition"))
  !(ntau isa Integer) && ntau != Inf && throw(ArgumentError("ntau must be an integer or Inf"))
  tau < oneunit(tau) && throw(ArgumentError("ntau must be an integer or Inf"))
  !isempty(additionaltime) &&
    any(x -> x isa Number || x < zero(x) || x > tau, additionaltime) &&
      throw(ArgumentError("additionaltime must be positive and be <= tau"))
  ismultidose = D <: NCADose
  time = nca.time
  conc = nca.conc
  dosetime  = ismultidose ? nca.dose.time : [d.time for d in nca.dose]
  doseinput = ismultidose ? nca.dose.amt  : [d.amt  for d in nca.dose]
  dosescaling = doseamount === nothing ? map(one, doseinput) : @. doseamount / doseinput
  tmptime = map(zero, time)
  map(v->append!(tmptime, v), [tau, dosetime, additionaltime])
  map(v->append!(tmptime, v), [@.((d + time) % tau) for d in time])
  # Check if all the input concentrations are 0, and if so, give that
  # simple answer.
  all(iszero, conc) && return DataFrame(conc=zero(eltype(conc)), tmptime)
  # Check if we will need to extrapolate
  tlast′ = tlast(nca)[end]
  if tau*ntau > tlast′
    λz = lambdaz(nca; recompute=false, kwargs...)
    clast′ = clast(nca; kwargs...)
    # check missing λz??
    # cannot continue extrapolating due to missing data (likely due to
    # half-life not calculable)
  else
    λz = nothing
  end
  currenttol = steadystatetol + oneunit(steadystatetol)
  taucount = 0
  prevconc = similar(conc)
  conc′    = deepcopy(conc)
  tmptime  = similar(time)
  atmp     = similar(conc, typeof(one(eltype(conc))))
  # Stop either for reaching steady-state or for reaching the requested number of doses
  while taucount < ntau && currenttol >= steadystatetol
    copyto!(prevconc, conc′)
    # Perform the dosing for a single dosing interval.
    for i in eachindex(dosetime)
      @. tmptime = time - dosetime[i] + (tau * taucount)
      # For the first dosing interval, make sure that we do not
      # assign concentrations before the dose is given.
      # Update the current concentration (previous concentration +
      # new concentration scaled by the relative dose)
      for i in eachindex(tmptime)
        tmptime′ = tmptime[i]
        tmptime[i] < zero(eltype(tmptime)) && continue
        dosescaling′ = dosescaling isa Number ? dosescaling : dosescaling[i] # handle scalar case
        conc′[i] = conc′[i] + dosescaling′ * interpextrapconc(nca, tmptime′; method=method, kwargs...)
      end
    end
    taucount = taucount + 1
    if any(iszero, conc′)
      # prevent division by 0. Since not all concentrations are 0,
      # all values will eventually be nonzero.
      currenttol = steadystatetol + oneunit(steadystatetol)
    else
      @. atmp = one(eltype(prevconc))-prevconc/conc′
      currenttol = norm(atmp, Inf)
    end
  end
  return DataFrame(conc=conc′, time=tmptime)
end
