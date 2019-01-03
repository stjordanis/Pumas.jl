"""
  clast(nca::NCASubject)

Calculate `clast`
"""
function clast(nca::NCASubject; pred=false, kwargs...)
  if pred
    λz, intercept, _, _ = lambdaz(nca; recompute=false, kwargs...)
    return exp(intercept - λz*tlast(nca))
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
function cmin(nca::NCASubject; kwargs...)
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
thalf(nca::NCASubject; kwargs...) = log(2)./lambdaz(nca; recompute=false, kwargs...)[1]

"""
  clf(nca::NCASubject; kwargs...)

Calculate total drug clearance divided by the bioavailability (F), which is just the
inverse of the dose normalizedosed ``AUC_0^\\inf``.
"""
function clf(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  if D === Nothing
    throw(ArgumentError("Dose must be known to compute CLF"))
  end
  map(inv, normalizedose(auc(nca; kwargs...), nca))
end

"""
  vss(nca::NCASubject; kwargs...)

Calculate apparent volume of distribution at equilibrium for IV bolus doses.
``V_{ss} = AUMC / {AUC_0^\\inf}^2`` for dose normalizedosed `AUMC` and `AUC`.
"""
function vss(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  if D === Nothing
    throw(ArgumentError("Dose must be known to compute V_ss"))
  end
  normalizedose(aumc(nca; kwargs...), nca) ./ (normalizedose(auc(nca; kwargs...), nca)).^2
end

"""
  vz(nca::NCASubject; kwargs...)

Calculate the volume of distribution during the terminal phase.
``V_z = 1/(AUC_0^\\inf\\lambda_z)`` for dose normalizedosed `AUC`.
"""
function vz(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  if D === Nothing
    throw(ArgumentError("Dose must be known to compute V_z"))
  end
  aucinf = normalizedose(auc(nca; kwargs...), nca)
  λ = lambdaz(nca; recompute=false, kwargs...)[1]
  @. inv(aucinf * λ)
end

"""
  bioav(nca::NCASubject, ithdose::Integer; kwargs...)

Bioavailability is the ratio of two AUC values.
``Bioavailability (F) = (AUC_0^\\infty_{po}/Dose_{po})/(AUC_0^\\infty_{iv}/Dose_{iv})``
"""
function bioav(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}, ithdose::Integer; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  multidose = D <: AbstractArray
  # if there is only a single dose
  multidose || return missing
  # if we only have IV or EV
  length(unique(getfield.(nca.dose, :formulation))) == 1 && return missing
  # initialize
  auc_0_inf_po = auc_0_inf_iv = zero(eltype(AUC))/oneunit(first(nca.dose).amt) # normalizedosed
  sol = zeros(typeof(auc_0_inf_po), axes(nca.dose))
  refdose = subject_at_ithdose(nca, ithdose)
  refauc  = normalizedose(auc(refdose; auctype=:inf, kwargs...), refdose)
  map(eachindex(nca.dose)) do idx
    subj = subject_at_ithdose(nca, idx)
    sol[idx] = normalizedose(auc(subj; auctype=:inf, kwargs...), subj)/refauc
  end
  return sol
end

"""
  cl(nca::NCASubject, ithdose=nothing; kwargs...)

Total drug clearance
"""
function cl(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}, ithdose=nothing; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  D === Nothing && throw(ArgumentError("Dose must be known to compute CL"))
  _clf = clf(nca; kwargs...)
  dose = nca.dose
  if D <: NCADose # single dose
    dose.formulation === IV || return missing
    _bioav = one(AUC)
    return _bioav*_clf
  else # multiple doses
    # TODO: check back on logic for this ithdose for CL, not sure about the user specification
    ithdose === nothing && throw(ArgumentError("`ithdose` must be provided for computing CL"))
    _bioav = bioav(nca, ithdose)
    map(eachindex(dose)) do idx
      subj = subject_at_ithdose(nca, idx)
      formulation = subj.dose.formulation
      if idx == ithdose
        formulation === IV || throw(ArgumentError("the formulation of `ithdose` must be IV"))
      end
      _bioav = formulation === IV ? one(eltype(AUC)) : _bioav
      _bioav*_clf[idx]
    end # end multidoses
  end # end if
end

"""
  tlag(nca::NCASubject; kwargs...)

The time prior to the first increase in concentration.
"""
function tlag(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  D === Nothing && throw(ArgumentError("Dose must be known to compute tlag"))
  nca.dose.formulation === IV && return missing
  idx = findfirst(c->c > nca.llq, nca.conc)
  return nca.time[idx]
end

"""
  mrt(nca::NCASubject; kwargs...)

Mean residence time from the time of dosing to the time of the last measurable
concentration.

IV infusion:
  ``AUMC/AUC - TI/2`` not implemented yet.
non-infusion:
  ``AUMC/AUC``
"""
function mrt(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  D === Nothing && throw(ArgumentError("Dose must be known to compute mrt"))
  aumc(nca; kwargs...) / auc(nca; kwargs...)
end

"""
  mat(nca::NCASubject; kwargs...)

Mean absorption time:
``MAT = MRT_po - MRT_iv``
"""
function mat(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  # dose is checked in `mrt`, so we don't need to check it in `mat`
  multidose = D <: AbstractArray
  multidose || error("Need more than one type of dose to calculate MAT")
  mrt_po = mrt_iv = zero(eltype(eltype(T)))
  for idx in eachindex(nca.dose)
    subj = subject_at_ithdose(nca, idx)
    if subj.dose.formulation === IV
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
function tau(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  D <: NCADose && return tlast(nca; kwargs...)-nca.dose.time
  dose = nca.dose
  return dose[end].time-dose[end-1].time
end

"""
  cavg(nca::NCASubject)

Average concentration over one period. ``C_{avg} = AUC_{tau}/Tau``. For multiple dosing only.
"""
function cavg(nca::NCASubject{C,T,AUC,AUMC,D,Z,F,N,I,P}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N,I,P}
  D <: NCADose && return auc(nca; auctype=:last, kwargs...)/tau(nca; kwargs...)
  subj = subject_at_ithdose(nca, 1)
  auc(subj; auctype=:last, kwargs...)/tau(nca; kwargs...)
end

"""
  fluctation(nca::NCASubject; kwargs...)

Peak trough fluctuation over one dosing interval at steady state.
``Fluctuation = 100*(C_{max} - C_{min})/C_{avg}``
"""
fluctation(nca::NCASubject; kwargs...) = 100*(cmax(nca) - cmin(nca))/cavg(nca; kwargs...)

"""
  accumulationindex(nca::NCASubject; kwargs...)

Theoretical accumulation ratio. ``Accumulation_index = 1/(1-exp(-Lambda_z*Tau))``.
"""
function accumulationindex(nca::NCASubject; kwargs...)
  tmp = -lambdaz(nca; recompute=false, kwargs...)[1]*tau(nca)
  inv(oneunit(tmp)-exp(tmp))
end

"""
  swing(nca::NCASubject; kwargs...)

  ``swing = (C_{max}-C_{min})/C_{min}``
"""
function swing(nca::NCASubject; kwargs...)
  _cmin = cmin(nca)
  (cmax(nca) - _cmin) ./ _cmin
end
