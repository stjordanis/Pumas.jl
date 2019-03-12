import DiffResults: DiffResult

Base.@pure flattentype(t) = NamedTuple{fieldnames(typeof(t)), NTuple{length(t), eltype(eltype(t))}}

abstract type LikelihoodApproximation end
struct FO <: LikelihoodApproximation end
struct FOI <: LikelihoodApproximation end
struct FOCE <: LikelihoodApproximation end
struct FOCEI <: LikelihoodApproximation end
struct Laplace <: LikelihoodApproximation end
struct LaplaceI <: LikelihoodApproximation end

"""
    _lpdf(d,x)

The log pdf: this differs from `Distributions.logdpf` definintion in a couple of ways:
- if `d` is a non-distribution it assumes the Dirac distribution.
- if `x` is `NaN` or `Missing`, it returns 0.
- if `d` is a `NamedTuple` of distributions, and `x` is a `NamedTuple` of observations, it computes the sum of the observed variables.
"""
_lpdf(d::Number, x::Number) = d == x ? 0.0 : -Inf
_lpdf(d::ConstDomain, x) = _lpdf(d.val, x)
_lpdf(d::Distributions.Sampleable,x::Missing) = zval(d)
_lpdf(d::Distributions.MultivariateDistribution,x::AbstractArray) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable,x::PDMat) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable,x::Number) = isnan(x) ? zval(d) : logpdf(d,x)
_lpdf(d::Constrained, x) = _lpdf(d.dist, x)
function _lpdf(ds::T, xs::S) where {T<:NamedTuple, S<:NamedTuple}
  sum(propertynames(xs)) do k
    haskey(ds, k) ? _lpdf(getproperty(ds, k), getproperty(xs, k)) : zero(getproperty(xs, k))
  end
end
_lpdf(d::Domain, x) = 0.0
function _lpdf(ds::AbstractVector, xs::AbstractVector)
  if length(ds) != length(xs)
    throw(DimensionMismatch("vectors must have same length"))
  end
  l = _lpdf(ds[1], xs[1])
  @inbounds for i in 2:length(ds)
    l += _lpdf(ds[i], xs[i])
  end
  return l
end

"""
    conditional_nll(m::PuMaSModel, subject::Subject, param, fixeffs, args...; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject` with parameters `param` and
random effects `fixeffs`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
conditional_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, args...; kwargs...) =
  first(conditional_nll_ext(m, subject, fixeffs, args...; kwargs...))

function conditional_nll_ext(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple,
                             randeffs::NamedTuple=sample_randeffs(m, fixeffs), args...; kwargs...)
  # Extract a vector of the time stamps for the observations
  obstimes = subject.time
  isnothing(obstimes) && throw(ArgumentError("no observations for subject"))

  # collate that arguments
  collated = m.pre(fixeffs, randeffs, subject)

  # create solution object
  solution = _solve(m, subject, collated, args...; kwargs...)

  if solution === nothing
     derived_dist = m.derived(collated, nothing, obstimes, subject)
  else
    # if solution contains NaN return Inf
    if any(isnan, solution[end]) || solution.retcode != :Success
      # FIXME! Do we need to make this type stable?
      return Inf, nothing
    end

    # extract distributions
    derived_dist = m.derived(collated, solution, obstimes, subject)
  end

  ll = _lpdf(derived_dist, subject.observations)
  return -ll, derived_dist
end

function conditional_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple, approx::Union{Laplace,FOCE}, args...; kwargs...)
  l, dist    = conditional_nll_ext(m, subject, fixeffs, randeffs, args...; kwargs...)
  if isinf(l)
    return l
  end
  l0, dist0 = conditional_nll_ext(m, subject, fixeffs, map(zero, randeffs), args...; kwargs...)
  conditional_nll(dist, dist0, subject)
end

# FIXME! Having both a method for randeffs::NamedTuple and vrandeffs::AbstractVector shouldn't really be necessary. Clean it up!
function conditional_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::Union{Laplace,FOCE}, args...; kwargs...)
  rfxset = m.random(fixeffs)
  randeffs = TransformVariables.transform(totransform(rfxset), vrandeffs)
  conditional_nll(m, subject, fixeffs, randeffs, approx, args...; kwargs...)
end

function conditional_nll(derived_dist, derived_dist0, subject)
  x = sum(propertynames(subject.observations)) do k
    sum(zip(getproperty(derived_dist,k),getproperty(derived_dist0,k),getproperty(subject.observations,k))) do (d,d0,obs)
      _lpdf(Normal(mean(d),sqrt(var(d0))), obs)
    end
  end
  return -x
end

"""
    penalized_conditional_nll(m::PuMaSModel, subject::Subject, param, fixeffs, args...; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as
[`conditional_nll`](@ref), except that it incorporates the penalty from the prior
distribution of the random effects.

Here `fixeffs` can be either a `NamedTuple` or a vector (representing a transformation of the
random effects to Cartesian space).
"""
function penalized_conditional_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple, args...;kwargs...)
  rfxset = m.random(fixeffs)
  conditional_nll(m, subject, fixeffs, randeffs, args...;kwargs...) - _lpdf(rfxset.params, randeffs)
end

function penalized_conditional_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, args...;kwargs...)
  rfxset = m.random(fixeffs)
  randeffs = TransformVariables.transform(totransform(rfxset), vrandeffs)
  conditional_nll(m, subject, fixeffs, randeffs, args...;kwargs...) - _lpdf(rfxset.params, randeffs)
end

function penalized_conditional_nll_fn(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, args...;kwargs...)
  y -> penalized_conditional_nll(m, subject, fixeffs, y, args...; kwargs...)
end



"""
    penalized_conditional_nll!(diffres::DiffResult, m::PuMaSModel, subject::Subject, param, fixeffs, args...;
                               hessian=diffres isa HessianResult, kwargs...)

Compute the penalized conditional negative log-likelihood (see [`penalized_conditional_nll`](@ref)),
storing the gradient (and optionally, the Hessian) wrt to `fixeffs` in `diffres`.
"""
function penalized_conditional_nll!(diffres::DiffResult, m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, args...;
                                    hessian=isa(diffres, DiffResult{2}), kwargs...)
  f = penalized_conditional_nll_fn(m, subject, fixeffs, args...; kwargs...)
  if hessian
    ForwardDiff.hessian!(diffres, f, vrandeffs)
  else
    ForwardDiff.gradient!(diffres, f, vrandeffs)
  end
end


"""
    randeffs_estimate(model, subject, param, approx, ...)

The point-estimate the random effects (being the mode of the empirical Bayes estimate) of a
particular subject at a particular parameter values. The result is returned as a vector
(transformed into Cartesian space).
"""
function randeffs_estimate(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, approx::LikelihoodApproximation, args...; kwargs...)
  rfxset = m.random(fixeffs)
  p = TransformVariables.dimension(totransform(rfxset))

  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(fixeffs), numtype(fixeffs))

  return randeffs_estimate(m, subject, fixeffs, zeros(T, p), approx, args...; kwargs...)
end

function randeffs_estimate(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::Union{LaplaceI,FOCEI}, args...; kwargs...)
  Optim.minimizer(Optim.optimize(penalized_conditional_nll_fn(m, subject, fixeffs, args...; kwargs...),
                                 vrandeffs,
                                 BFGS(linesearch=Optim.LineSearches.BackTracking());
                                 autodiff=:forward))
end

function randeffs_estimate(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::Union{Laplace,FOCE}, args...; kwargs...)
  Optim.minimizer(Optim.optimize(s -> penalized_conditional_nll(m, subject, fixeffs, (η=s,), approx, args...; kwargs...),
                                 vrandeffs,
                                 BFGS(linesearch=Optim.LineSearches.BackTracking());
                                 autodiff=:forward))
end

randeffs_estimate(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, ::Union{FO,FOI}, args...; kwargs...) = zero(vrandeffs)

"""
    randeffs_estimate_dist(model, subject, param, approx, ...)

Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a vector (transformed into Cartesian space).
"""
function randeffs_estimate_dist(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, approx::LaplaceI, args...; kwargs...)
  vrandeffs = randeffs_estimate(m, subject, fixeffs, approx, args...; kwargs...)
  diffres = DiffResults.HessianResult(vrandeffs)
  penalized_conditional_nll!(diffres, m, subject, fixeffs, vrandeffs, args...; hessian=true, kwargs...)
  MvNormal(vrandeffs, Symmetric(inv(DiffResults.hessian(diffres))))
end

"""
    marginal_nll(model, subject, param[, fixeffs], approx, ...)
    marginal_nll(model, population, param, approx, ...)

Compute the marginal negative loglikelihood of a subject or dataset, using the integral
approximation `approx`. If no random effect (`fixeffs`) is provided, then this is estimated
from the data.

See also [`marginal_nll_nonmem`](@ref).
"""
function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple, approx::LikelihoodApproximation, args...;
                      kwargs...)
  rfxset = m.random(fixeffs)
  vrandeffs = TransformVariables.inverse(totransform(rfxset), randeffs)
  marginal_nll(m, subject, fixeffs, vrandeffs, approx, args...; kwargs...)
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::LaplaceI, args...;
                      kwargs...)
  diffres = DiffResults.HessianResult(vrandeffs)
  penalized_conditional_nll!(diffres, m, subject, fixeffs, vrandeffs, args...; hessian=true, kwargs...)
  g, m, W = DiffResults.value(diffres),DiffResults.gradient(diffres),DiffResults.hessian(diffres)
  CW = cholesky!(Symmetric(W), check=false) # W is positive-definite, only compute Cholesky once.
  if issuccess(CW)
    p = length(vrandeffs)
    g - (p*log(2π) - logdet(CW) + dot(m,CW\m))/2
  else
    return typeof(g)(Inf)
  end
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, ::FOCEI, args...; kwargs...)

  # The dimension of the random effect vector
  nrfx = length(vrandeffs)

  # Construct vector of dual numbers for the random effects to track the partial derivatives
  vrandeffsdual = [ForwardDiff.Dual{typeof(ForwardDiff.Tag(conditional_nll_ext,eltype(vrandeffs)))}(vrandeffs[j], ntuple(i -> eltype(vrandeffs)(i == j), nrfx)...) for j in 1:nrfx]

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  nl_d, dist_d = conditional_nll_ext(m, subject, fixeffs, (η=vrandeffsdual,), args...; kwargs...)

  nl = ForwardDiff.value(nl_d)

  if isfinite(nl)
    # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
    Ω = cov(m.random(fixeffs).params.η)
    FΩ = cholesky(Ω, check=false)

    # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
    if issuccess(FΩ)
      w = FIM(dist_d, FOCEI())
      nl += (logdet(Ω) + vrandeffs'*(FΩ\vrandeffs) + logdet(inv(FΩ) + w))/2
    else # Ω is numerically singular
      nl = typeof(nl)(Inf)
    end
  end
  return nl
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, ::FOCE, args...; kwargs...)

  # The dimension of the random effect vector
  nrfx = length(vrandeffs)

  # Construct vector of dual numbers for the random effects to track the partial derivatives
  vrandeffsdual = [ForwardDiff.Dual{typeof(ForwardDiff.Tag(conditional_nll_ext,eltype(vrandeffs)))}(vrandeffs[j], ntuple(i -> eltype(vrandeffs)(i == j), nrfx)...) for j in 1:nrfx]

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  nl_d, dist_d = conditional_nll_ext(m, subject, fixeffs, (η=vrandeffsdual,), args...; kwargs...)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation for η=0
  nl_0, dist_0 = conditional_nll_ext(m, subject, fixeffs, (η=zero(vrandeffs),), args...; kwargs...)

  # Extract the value of the conditional likelihood
  nl = ForwardDiff.value(conditional_nll(dist_d, dist_0, subject))

  # Compute the Hessian approxmation in the random effect vector η
  W = FIM(dist_d, dist_0, FOCE())

  # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
  Ω = cov(m.random(fixeffs).params.η)
  FΩ = cholesky(Ω, check=false)

  # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
  if issuccess(FΩ)
    return nl + (logdet(FΩ) + vrandeffs'*(FΩ\vrandeffs) + logdet(inv(FΩ) + W))/2
  else # Ω is numerically singular
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, ::FO, args...; kwargs...)

  # For FO, the conditional likelihood must be evaluated at η=0
  @assert iszero(vrandeffs)

  # The dimension of the random effect vector
  nrfx = length(vrandeffs)

  # Construct vector of dual numbers for the random effects to track the partial derivatives
  vrandeffsdual = [ForwardDiff.Dual{typeof(ForwardDiff.Tag(conditional_nll_ext,eltype(vrandeffs)))}(vrandeffs[j], ntuple(i -> eltype(vrandeffs)(i == j), nrfx)...) for j in 1:nrfx]

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  l_d, dist_d = conditional_nll_ext(m, subject, fixeffs, (η=vrandeffsdual,), args...; kwargs...)

  # Extract the value of the conditional likelihood
  l = ForwardDiff.value(l_d)

  # Compute the gradient of the likelihood and Hessian approxmation in the random effect vector η
  W, dldη = FIM(subject.observations, dist_d, FO())

  # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
  Ω = cov(m.random(fixeffs).params.η)
  FΩ = cholesky(Ω, check=false)

  # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
  if issuccess(FΩ)
    invFΩ = inv(FΩ)
    return l + (logdet(FΩ) - dldη'*((invFΩ + W)\dldη) + logdet(invFΩ + W))/2
  else # Ω is numerically singular
    return typeof(l)(Inf)
  end
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::Laplace, args...; kwargs...)
  Ω = cov(m.random(fixeffs).params.η)
  nl = conditional_nll(m, subject, fixeffs, vrandeffs, approx, args...; kwargs...)
  W = ForwardDiff.hessian(s -> conditional_nll(m, subject, fixeffs, s, approx, args...; kwargs...), vrandeffs)
  return nl + (logdet(Ω) + vrandeffs'*(Ω\vrandeffs) + logdet(inv(Ω) + W))/2
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, approx::LikelihoodApproximation, args...;
                      kwargs...)
  vrandeffs = randeffs_estimate(m, subject, fixeffs, approx, args...; kwargs...)
  marginal_nll(m, subject, fixeffs, vrandeffs, approx, args...; kwargs...)
end
function marginal_nll(m::PuMaSModel, data::Population, args...;
                      kwargs...)
  sum(subject -> marginal_nll(m, subject, args...; kwargs...), data.subjects)
end


"""
    marginal_nll_nonmem(model, subject, param[, fixeffs], approx, ...)
    marginal_nll_nonmem(model, data, param, approx, ...)

Compute the NONMEM-equivalent marginal negative loglikelihood of a subject or dataset:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
marginal_nll_nonmem(m::PuMaSModel, subject::Subject, args...; kwargs...) =
    2marginal_nll(m, subject, args...; kwargs...) - length(first(subject.observations))*log(2π)
marginal_nll_nonmem(m::PuMaSModel, data::Population, args...; kwargs...) =
    2marginal_nll(m, data, args...; kwargs...) - sum(subject->length(first(subject.observations)), data.subjects)*log(2π)
# NONMEM doesn't allow ragged, so this suffices for testing


"""
In named tuple nt, replace the value x.var by y
"""
@generated function Base.setindex(x::NamedTuple,y,v::Val)
  k = first(v.parameters)
  k ∉ x.names ? :x : :( (x..., $k=y) )
end

function _mean(model, subject, fixeffs::NamedTuple, randeffs::NamedTuple, i, args...; kwargs...)
  x_, dist_ = conditional_nll_ext(model, subject, fixeffs, randeffs, args...; kwargs...)
  mean(dist_.dv[i])
end

function FIM(dist::NamedTuple, ::FOCEI)

  # FIXME! Currently we hardcode for dv. Eventually, this should allow for arbitrary dependent variables
  dv  = dist.dv

  # Loop through the distribution vector and extract derivative information
  H = sum(1:length(dv)) do j
    dvj   = dv[j]
    r_inv = inv(ForwardDiff.value(var(dvj)))
    f     = SVector(ForwardDiff.partials(mean(dvj)).values)
    del_r = SVector(ForwardDiff.partials(var(dvj)).values)
    f*r_inv*f' + (r_inv*del_r*r_inv*del_r')/2
  end

  return H
end

function FIM(dist::NamedTuple, dist0::NamedTuple, ::FOCE)
  # FIXME! Currently we hardcode for dv. Eventually, this should allow for arbitrary dependent variables
  dv  = dist.dv
  dv0 = dist0.dv

  # Loop through the distribution vector and extract derivative information
  H = sum(1:length(dv)) do j
    r_inv = inv(var(dv0[j]))
    f = SVector(ForwardDiff.partials(mean(dv[j])).values)
    f*r_inv*f'
  end

  return H
end

function FIM(obs::NamedTuple, dist::NamedTuple, ::FO)
  # FIXME! Currently we hardcode for dv. Eventually, this should allow for arbitrary dependent variables
  dv = dist.dv

  # The dimension of the random effect vector
  nrfx = length(ForwardDiff.partials(mean(dv[1])))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)

  # Loop through the distribution vector and extract derivative information
  for j in 1:length(dv)
    dvj = dv[j]
    r = ForwardDiff.value(var(dvj))
    f = SVector(ForwardDiff.partials(mean(dvj)).values)
    fdr = f/r
    H += fdr*f'
    dldη += fdr*(obs.dv[j] - ForwardDiff.value(mean(dvj)))
  end

  return H, dldη
end

# Fitting methods
struct FittedPuMaSModel{T1<:PuMaSModel,T2<:Population,T3,T4<:LikelihoodApproximation}
  model::T1
  data::T2
  optim::T3
  approx::T4
end

function Distributions.fit(m::PuMaSModel,
                           data::Population,
                           fixeffs::NamedTuple,
                           approx::LikelihoodApproximation,
                           args...;
                           verbose=false,
                           optimmethod::Optim.AbstractOptimizer=BFGS(linesearch=Optim.LineSearches.BackTracking()),
                           optimautodiff=:finite,
                           optimoptions::Optim.Options=Optim.Options(show_trace=verbose, # Print progress
                                                                     g_tol=1e-3),
                           kwargs...)
  trf = totransform(m.param)
  vfixeffs = TransformVariables.inverse(trf, fixeffs)
  o = optimize(s -> marginal_nll(m, data, TransformVariables.transform(trf, s), approx, args...; kwargs...),
               vfixeffs, optimmethod, optimoptions, autodiff=optimautodiff)

  return FittedPuMaSModel(m, data, o, approx)
end

function Base.getproperty(f::FittedPuMaSModel{<:Any,<:Any,<:Optim.MultivariateOptimizationResults}, s::Symbol)
  if s === :fixeffs
    trf = totransform(f.model.param)
    TransformVariables.transform(trf, f.optim.minimizer)
  else
    return getfield(f, s)
  end
end

marginal_nll(       f::FittedPuMaSModel) = marginal_nll(       f.model, f.data, f.fixeffs, f.approx)
marginal_nll_nonmem(f::FittedPuMaSModel) = marginal_nll_nonmem(f.model, f.data, f.fixeffs, f.approx)
