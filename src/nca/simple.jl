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
function ctlast_idx(conc, time; llq=nothing)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findlast(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

function ctfirst_idx(conc, time; llq=nothing)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findfirst(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

"""
    tmax(nca::NCASubject; interval=nothing, kwargs...)

Calculate ``T_{max}_{t_1}^{t_2}``
"""
function tmax(nca::NCASubject; interval=nothing, kwargs...)
  if interval isa Union{Tuple, Nothing}
    return ctextreme(nca; interval=interval, kwargs...)[2]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctextreme(nca; interval=i, kwargs...)[2]
    end
    map(f, interval)
  end
end

"""
    cmax(nca::NCASubject; normalize=false, interval=nothing, kwargs...)

Calculate ``C_{max}_{t_1}^{t_2}``
"""
function cmax(nca::NCASubject; normalize=false, kwargs...)
  dose = nca.dose
  cmax′ = _cmax(nca; kwargs...)
  if dose !== nothing && dose.ss
    return cmax′/accumulationindex(nca)
  else
    return cmax′
  end
end

"""
    cmaxss(nca::NCASubject; normalize=false, kwargs...)

Calculate ``C_{maxss}``
"""
function cmaxss(nca::NCASubject; interval=nothing, kwargs...)
  dose = nca.dose
  cmax′ = _cmax(nca; kwargs...)
  if dose === nothing || (!dose.ss && dose.ii > zero(dose.ii))
    return cmax′*accumulationindex(nca)
  else # SS, so `cmax′` is actually `cmax_ss`
    return cmax′
  end
end

function _cmax(nca::NCASubject; interval=nothing, normalize=false, kwargs...)
  if interval isa Union{Tuple, Nothing}
    sol = ctextreme(nca; interval=interval, kwargs...)[1]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctextreme(nca; interval=i, kwargs...)[1]
    end
    sol = map(f, interval)
  end
  normalize ? normalizedose(sol, nca) : sol
end

"""
    ctau(nca::NCASubject; method=:linear, kwargs...)

Calculate concentration at τ
"""
ctau(nca::NCASubject; method=:linear, kwargs...) = interpextrapconc(nca, tau(nca; kwargs...); method=method, kwargs...)

"""
    cmin(nca::NCASubject; normalize=false, interval=nothing, kwargs...)

Calculate ``C_{min}_{t_1}^{t_2}``
"""
function cmin(nca::NCASubject; normalize=false, kwargs...)
  dose = nca.dose
  cmin′ = _cmin(nca; kwargs...)
  if dose !== nothing && dose.ss
    return cmin′/accumulationindex(nca)
  else
    return cmin′
  end
end

"""
    cminss(nca::NCASubject; normalize=false, kwargs...)

Calculate ``C_{minss}``
"""
function cminss(nca::NCASubject; interval=nothing, kwargs...)
  dose = nca.dose
  cmin′ = _cmin(nca; kwargs...)
  if dose === nothing || (!dose.ss && dose.ii > zero(dose.ii))
    return cmin′*accumulationindex(nca)
  else # SS, so `cmin′` is actually `cmin_ss`
    return cmin′
  end
end

function _cmin(nca::NCASubject; kwargs...)
  (nca.dose !== nothing && nca.dose.formulation === EV && !nca.dose.ss) && return missing
  val = ctextreme(nca, >)[1]
  return val
end

"""
    tmin(nca::ncasubject; kwargs...)

Calculate time of minimum observed concentration
"""
function tmin(nca::NCASubject; kwargs...)
  val = ctextreme(nca, >)[2]
  return val
end

function ctextreme(nca::NCASubject, lt=<; interval=nothing, kwargs...)
  conc, time = nca.conc, nca.time
  c0′ = (interval === nothing || iszero(interval[1])) ? c0(nca, true) : missing
  if interval === nothing
    val, idx = conc_extreme(conc, eachindex(conc), lt)
    t = time[idx]
    if !ismissing(c0′)
      val, t, idx = lt(val, c0′) ? (c0′, zero(t), -1) : (val, t, idx) # calculate max by default
    end
    return (val, t, idx)
  elseif interval isa Tuple
    @assert interval[1] < interval[2] "t0 must be less than t1"
    interval[1] > time[end] && throw(ArgumentError("t0 is longer than observation time"))
    idx1, idx2 = let (lo, hi)=interval
      findfirst(t->t>=lo, time),
      findlast( t->t<=hi, time)
    end
    val, idx = conc_extreme(conc, idx1:idx2, lt)
    t = time[idx]
    if !ismissing(c0′)
      val, t, idx = lt(val, c0′) ? (c0′, zero(t), -1) : (val, t, idx) # calculate max by default
    end
  else
    throw(ArgumentError("interval must be nothing or a tuple."))
  end
  return (val, t, idx)
end

maxidx(subj::NCASubject) = subj.maxidx == -2 ? (subj.maxidx = ctextreme(subj)[3]) : subj.maxidx

function conc_extreme(conc, idxs, lt)
  local val, idx
  for i in idxs
    conci = conc[i]
    !ismissing(conci) && (val = conci; idx=i; break) # find a sensible initialization
  end
  for i in idxs
    if !ismissing(conc[i])
      lt(val, conc[i]) && (val = conc[i]; idx=i)
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
    cl(nca::NCASubject; kwargs...)

Calculate total drug clearance.
"""
function cl(nca::NCASubject; auctype=nothing, kwargs...)
  nca.dose === nothing && return missing
  if nca.dose.ss
    1/normalizedose(auctau(nca; kwargs...), nca)
  else
    1/normalizedose(auc(nca; kwargs...), nca)
  end
end

function _clf(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation === EV ? cl(nca; kwargs...) : missing
end
function _cl(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation !== EV ? cl(nca; kwargs...) : missing
end

"""
    vss(nca::NCASubject; kwargs...)

Calculate apparent volume of distribution at equilibrium for IV bolus doses.
``V_{ss} = MRT * CL``.
"""
vss(nca::NCASubject; kwargs...) = mrt(nca; kwargs...)*cl(nca; kwargs...)

"""
    vz(nca::NCASubject; kwargs...)

Calculate the volume of distribution during the terminal phase.
``V_z = 1/(AUC⋅λz)`` for dose normalizedosed `AUC`.
"""
function vz(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  λ = lambdaz(nca; recompute=false, kwargs...)
  if nca.dose.ss
    τauc = auctau(nca; kwargs...)
    @. 1/(normalizedose(τauc, nca.dose) * λ)
  else
    aucinf = normalizedose(auc(nca; kwargs...), nca)
    @. 1/(aucinf * λ)
  end
end

function _vzf(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation === EV ? vz(nca; kwargs...) : missing
end
function _vz(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation !== EV ? vz(nca; kwargs...) : missing
end

"""
    bioav(nca::NCASubject; ithdose::Integer, kwargs...)

Bioavailability is the ratio of two AUC values.
``Bioavailability (F) = (AUC_0^\\infty_{po}/Dose_{po})/(AUC_0^\\infty_{iv}/Dose_{iv})``
"""
function bioav(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}; ithdose=missing, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
  n = nca.dose isa AbstractArray ? length(nca.dose) : 1
  ismissing(ithdose) && return n == 1 ? missing : fill(missing, n)
  multidose = n > 1
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

#isiv(x) = x === IVBolus || x === IVInfusion
#"""
#    cl(nca::NCASubject; ithdose::Integer, kwargs...)
#
#Total drug clearance
#"""
#function cl(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}; ithdose=missing, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
#  nca.dose === nothing && return missing
#  _clf = clf(nca; kwargs...)
#  @show 1
#  dose = nca.dose
#  if nca.dose isa NCADose # single dose
#    isiv(dose.formulation) || return missing
#    _bioav = one(eltype(AUC))
#    return _bioav*_clf
#  else # multiple doses
#    # TODO: check back on logic for this ithdose for CL, not sure about the user specification
#    ismissing(ithdose) && return missing
#    map(eachindex(dose)) do idx
#      subj = subject_at_ithdose(nca, idx)
#      formulation = subj.dose.formulation
#      if idx == ithdose
#        isiv(formulation) || throw(ArgumentError("the formulation of `ithdose` must be IV"))
#      end
#      if isiv(formulation)
#        one(eltype(AUC))*_clf[idx]
#      else
#        missing
#      end
#    end # end multidoses
#  end # end if
#end

"""
    tlag(nca::NCASubject; kwargs...)

The time prior to the first increase in concentration.
"""
function tlag(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  (nca.rate === nothing && nca.dose.formulation !== EV) && return missing
  idx = findfirst(c->c > nca.llq, nca.conc)-1
  return 1 <= idx <= length(nca.time) ? nca.time[idx] : zero(nca.time[1])
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
function mrt(nca::NCASubject; auctype=:inf, kwargs...)
  dose = nca.dose
  dose === nothing && return missing
  ti2 = dose.duration*1//2
  if dose.ss
    auctype === :last && return missing
    τ = tau(nca; kwargs...)
    aumcτ = aumctau(nca; kwargs...)
    aucτ = auctau(nca; kwargs...)
    _auc = auc(nca; auctype=:inf, kwargs...)
    quotient = (aumcτ + τ*(_auc - aucτ)) / aucτ
    dose.formulation === IVInfusion && (quotient -= ti2)
    return quotient
  else
    quotient = aumc(nca; auctype=auctype, kwargs...) / auc(nca; auctype=auctype, kwargs...)
    dose.formulation === IVInfusion && (quotient -= ti2)
    return quotient
  end
end

# Not possible to calculate MAT
#"""
#    mat(nca::NCASubject; kwargs...)
#
#Mean absorption time:
#``MAT = MRT_po - MRT_iv``
#
#For multiple dosing only.
#"""
#function mat(nca::NCASubject; kwargs...)
#  # dose is checked in `mrt`, so we don't need to check it in `mat`
#  multidose = nca.dose isa AbstractArray
#  #multidose || error("Need more than one type of dose to calculate MAT")
#  multidose || return missing
#  mrt_po = mrt_iv = zero(eltype(eltype(nca.time)))
#  for idx in eachindex(nca.dose)
#    subj = subject_at_ithdose(nca, idx)
#    if isiv(subj.dose.formulation)
#      mrt_iv += mrt(subj; kwargs...)
#    else
#      mrt_po += mrt(subj; kwargs...)
#    end
#  end # end for
#  mrt_po - mrt_iv
#end

"""
    tau(nca::NCASubject)

Dosing interval. For multiple dosing only.
"""
function tau(nca::NCASubject; kwargs...)
  nca.dose !== nothing || return missing
  !iszero(nca.dose.ii) && return nca.dose.ii
  return missing
  # do not guess tau
  #dose = nca.dose
  #(dose === nothing || AUC isa Number) && return missing
  #dose isa NCADose && return nca.abstime[nca.lastidx]-dose.time
  #return dose[end].time-dose[end-1].time
end

"""
    cavgss(nca::NCASubject)

Average concentration over one period. ``C_{avgss} = AUC_{tau}/Tau``.
"""
function cavgss(nca::NCASubject; pred=false, kwargs...)
  nca.dose === nothing && return missing
  subj = nca.dose isa NCADose ? nca : subject_at_ithdose(nca, 1)
  ii = tau(subj)
  if nca.dose.ss
    return auctau(subj; kwargs...)/ii
  else
    return auc(subj; auctype=:inf, pred=pred)/ii
  end
end

"""
    fluctuation(nca::NCASubject; usetau=false, kwargs...)

Peak trough fluctuation over one dosing interval at steady state.
``Fluctuation = 100*(C_{maxss} - C_{minss})/C_{avgss}`` (usetau=false) or
``Fluctuation = 100*(C_{maxss} - C_{tau})/C_{avgss}`` (usetau=true)
"""
function fluctuation(nca::NCASubject; usetau=false, kwargs...)
  _cmin = usetau ? ctau(nca) : cminss(nca)
  100*(cmaxss(nca) - _cmin)/cavgss(nca; kwargs...)
end

"""
    accumulationindex(nca::NCASubject; kwargs...)

Theoretical accumulation ratio. ``Accumulation_index = 1/(1-exp(-λ_z*Tau))``.
"""
function accumulationindex(nca::NCASubject; kwargs...)
  tmp = -lambdaz(nca; recompute=false, kwargs...)*tau(nca)
  one(tmp)/(one(tmp)-exp(tmp))
end

"""
    swing(nca::NCASubject; usetau=false, kwargs...)

``swing = (C_{maxss}-C_{minss})/C_{minss}`` (usetau=false) or ``swing = (C_{maxss}-C_{tau})/C_{tau}`` (usetau=true)
"""
function swing(nca::NCASubject; usetau=false, kwargs...)
  _cmin = usetau ? ctau(nca) : cminss(nca)
  sw = (cmaxss(nca) - _cmin) ./ _cmin
  (sw === missing || isinf(sw)) ? missing : sw
end

"""
    c0(nca::NCASubject; kwargs...)

Estimate the concentration at dosing time for an IV bolus dose.
"""
function c0(subj::NCASubject, returnev=false; verbose=true, kwargs...) # `returnev` is not intended to be used by users
  subj.dose === nothing && return missing
  t1 = ustrip(subj.time[1])
  isurine = subj.rate !== nothing
  if subj.dose.formulation !== IVBolus || isurine
    return !returnev ? missing :
               (isurine || !subj.dose.ss) ? zero(subj.conc[1]) :
               ctau(subj; kwargs...)
  end
  iszero(t1) && return subj.conc[1]
  t2 = ustrip(subj.time[2])
  c1 = ustrip(subj.conc[1]); c2 = ustrip(subj.conc[2])
  iszero(c1) && return c1
  if c2 >= c1 && verbose
    @warn "c0: This is an IV bolus dose, but the first two concentrations are not decreasing. If `conc[i]/conc[i+1] > 0.8` holds, the back extrapolation will be computed internally for AUC and AUMC, but will not be reported."
  end
  if c2 < c1 || (returnev && c1/c2 > 0.8)
    dosetime = ustrip(subj.dose.time)
    c0 = exp(log(c1) - (t1 - dosetime)*(log(c2)-log(c1))/(t2-t1))*oneunit(eltype(subj.conc))
  else
    c0 = missing
  end
  return c0
end

#= issue #391
# The function is originally translated from the R package PKNCA
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
=#

# TODO: user input lambdaz, clast, and tlast?
# TODO: multidose?
function superposition(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R},
                       tau::Number, ntau=Inf, args...;
                       doseamount=nothing, additionaltime::Vector=T[],
                       steadystatetol::Number=1e-3, method=:linear,
                       kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
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

n_samples(subj::NCASubject; kwargs...) = length(subj.time)
doseamt(subj::NCASubject; kwargs...) = hasdose(subj) ? subj.dose.amt : missing
dosetype(subj::NCASubject; kwargs...) = hasdose(subj) ? string(subj.dose.formulation) : missing

urine_volume(subj::NCASubject; kwargs...) = subj.volume === nothing ? missing : sum(subj.volume)
amount_recovered(subj::NCASubject; kwargs...) = sum(prod, zip(subj.conc, subj.volume))
function percent_recovered(subj::NCASubject; kwargs...)
  subj.dose === nothing && return missing
  sum(amount_recovered(subj) / subj.dose.amt) * 100
end
