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
                             randeffs::NamedTuple=rand_random(m, fixeffs), args...; kwargs...)
  # Extract a vector of the time stamps for the observations
  obstimes = subject.time
  isnothing(obstimes) && throw(ArgumentError("no observations for subject"))

  # collate that arguments
  collated = m.pre(fixeffs, randeffs, subject.covariates)

  # create solution object
  solution = _solve(m, subject, collated, args...; kwargs...)

  if solution === nothing
     derived_dist = m.derived(collated, nothing, obstimes)
  else
    # compute solution values
    path = solution.(obstimes, continuity=:right)

    # if solution contains NaN return Inf
    if any(isnan, last(path)) || solution.retcode != :Success
      # FIXME! Do we need to make this type stable?
      return Inf, nothing
    end

    # extract distributions
    derived_dist = m.derived(collated, path, obstimes)
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
randeffs_estimate

function randeffs_estimate(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, approx::Union{LaplaceI,FOCEI}, args...; kwargs...)
  rfxset = m.random(fixeffs)
  p = TransformVariables.dimension(totransform(rfxset))
  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(fixeffs), numtype(fixeffs))
  Optim.minimizer(Optim.optimize(penalized_conditional_nll_fn(m, subject, fixeffs, args...; kwargs...), zeros(T, p), BFGS(); autodiff=:forward))
end

function randeffs_estimate(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, approx::Union{Laplace,FOCE}, args...; kwargs...)
  rfxset = m.random(fixeffs)
  p = TransformVariables.dimension(totransform(rfxset))
  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(fixeffs), numtype(fixeffs))
  Optim.minimizer(Optim.optimize(s -> penalized_conditional_nll(m, subject, fixeffs, (η=s,), approx, args...; kwargs...), zeros(T, p), BFGS(); autodiff=:forward))
end

function randeffs_estimate(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, approx::Union{FO,FOI}, args...; kwargs...)
  rfxset = m.random(fixeffs)
  p = TransformVariables.dimension(totransform(rfxset))
  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(fixeffs), numtype(fixeffs))
  return zeros(T, p)
end

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
  CW = cholesky!(Symmetric(W)) # W is positive-definite, only compute Cholesky once.
  p = length(vrandeffs)
  g - (p*log(2π) - logdet(CW) + dot(m,CW\m))/2
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::FOCEI, args...; kwargs...)
  Ω = cov(m.random(fixeffs).params.η)
  l,dist = conditional_nll_ext(m, subject, fixeffs, (η=vrandeffs,), args...; kwargs...)
  w = FIM(m, subject, fixeffs, vrandeffs, approx, dist, args...; kwargs...)
  l += (logdet(Ω) + vrandeffs'*(Ω\vrandeffs) + logdet(inv(Ω) + w))/2
  return l
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::FOCE, args...; kwargs...)
  Ω = cov(m.random(fixeffs).params.η)
  nl = conditional_nll(m, subject, fixeffs, vrandeffs, approx, args...; kwargs...)
  W = FIM(m,subject, fixeffs, vrandeffs, approx, args...; kwargs...)

  FΩ = cholesky(Ω, check=false)
  if issuccess((FΩ))
    return nl + (logdet(FΩ) + vrandeffs'*(FΩ\vrandeffs) + logdet(inv(FΩ) + W))/2
  else # Ω is numerically singular
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::FO, args...; kwargs...)
  Ω = cov(m.random(fixeffs).params.η)
  l0, dist0 = conditional_nll_ext(m, subject, fixeffs, (η=zero(vrandeffs),), args...; kwargs...)
  W, dldη = FIM(m, subject, fixeffs, zero(vrandeffs), approx, dist0, args...; kwargs...)

  FΩ = cholesky(Ω, check=false)
  if issuccess((FΩ))
    invFΩ = inv(FΩ)
    return l0 + (logdet(FΩ) - dldη'*((invFΩ + W)\dldη) + logdet(invFΩ + W))/2
  else # Ω is numerically singular
    return typeof(l0)(Inf)
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

function generate_enclosed_likelihood(ll, model::PuMaSModel, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple, v, args...; kwargs...)
  function (z)
    _fixeffs = setindex(fixeffs,z,v)
    _randeffs = setindex(randeffs,z,v)
    ll(model, subject, _fixeffs, _randeffs, args...; kwargs...)
  end
end

function ll_derivatives(ll, model::PuMaSModel, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple, var::Symbol, args...; kwargs...)
  ll_derivatives(ll, model, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple, Val(var), args...; kwargs...)
end

@generated function ll_derivatives(ll, model::PuMaSModel, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple,
                                   v::Val{var}, args...;
                                   hessian_required = true,
                                   transform=false, kwargs...) where var
  var ∉ fieldnames(fixeffs) ∪ fieldnames(randeffs) && error("Variable name not recognized")
  valex = var ∈ fieldnames(fixeffs) ? :(fixeffs.$var) : :(randeffs.$var)
  quote
    f = generate_enclosed_likelihood(ll,model,subject,fixeffs,randeffs,v, args...; kwargs...)
    if hessian_required
      result = DiffResults.HessianResult($valex)
      result = ForwardDiff.hessian!(result, f, $valex)
      return result
    else
      result = DiffResults.GradientResult($valex)
      result = ForwardDiff.gradient!(result, f, $valex)
      return result
    end
  end
end

function _mean(model, subject, fixeffs::NamedTuple, randeffs::NamedTuple, i, args...; kwargs...)
  x_, dist_ = conditional_nll_ext(model, subject, fixeffs, randeffs, args...; kwargs...)
  mean(dist_.dv[i])
end

function _var(model, subject, fixeffs::NamedTuple, randeffs::NamedTuple, i, args...; kwargs...)
  x_, dist_ = conditional_nll_ext(model, subject, fixeffs, randeffs, args...; kwargs...)
  var(dist_.dv[i])
end

function _mean0(model, subject, fixeffs::NamedTuple, randeffs::NamedTuple, i, args...; kwargs...)
  _mean(model, subject, fixeffs, map(zero, randeffs), i, args...; kwargs...)
end

function FIM(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::FOCEI, dist, args...; kwargs...)
  f_res     = DiffResults.GradientResult(vrandeffs)
  del_r_res = DiffResults.GradientResult(vrandeffs)
  fim = sum(1:length(subject.observations.dv)) do j
    r_inv = inv(var(dist.dv[j]))
    # FIXME! Proper names should be kept and passed along
    ForwardDiff.gradient!(f_res, s -> _mean(m, subject, fixeffs, (η=s,), j, args...; kwargs...), vrandeffs)
    f = DiffResults.gradient(f_res)
    # FIXME! Proper names should be kept and passed along
    ForwardDiff.gradient!(del_r_res, s -> _var(m, subject, fixeffs, (η=s,), j, args...; kwargs...), vrandeffs)
    del_r = DiffResults.gradient(del_r_res)
    return f*r_inv*f' + (r_inv*del_r*r_inv*del_r')/2
  end
  fim
end

function FIM(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::FOCE, args...; kwargs...)
  l0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),), args...; kwargs...)
  fim = sum(1:length(subject.observations.dv)) do j
    r_inv = inv(var(dist0.dv[j]))
    # FIXME! Proper names should be kept and passed along
    res = ll_derivatives(_mean, m, subject, fixeffs, (η=vrandeffs,), :η, j, args...; kwargs...)
    f = DiffResults.gradient(res)
    f*r_inv*f'
  end
  fim
end

function FIM(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector, approx::FO, dist0, args...; kwargs...)
  r = var(dist0.dv[1])
  f = ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), 1, args...; kwargs...), vrandeffs)
  fdr = f/r
  H = fdr*f'
  mean0 = _mean0(m, subject, fixeffs, (η=vrandeffs,), 1, args...; kwargs...)
  # FIXME! This is not correct when there are more than a single dependent variable
  dldη = fdr*(subject.observations.dv[1] - mean0)

  for j in 2:length(subject.observations.dv)
    r = var(dist0.dv[j])
    ForwardDiff.gradient!(f, s -> _mean(m, subject, fixeffs, (η=s,), j, args...; kwargs...), vrandeffs)
    fdr = f/r
    H += fdr*f'
    mean0 = _mean0(m, subject, fixeffs, (η=vrandeffs,), j, args...; kwargs...)
    # FIXME! This is not correct when there are more than a single dependent variable
    dldη += fdr*(subject.observations.dv[j] - mean0)
  end

  return H, dldη
end

# Fitting methods
struct FittedPuMaSModel{T1<:PuMaSModel,T2<:Population,T3<:NamedTuple,T4<:LikelihoodApproximation}
  model::T1
  data::T2
  fixeffs::T3
  approx::T4
end

function Distributions.fit(m::PuMaSModel,
                           data::Population,
                           fixeffs::NamedTuple,
                           approx::LikelihoodApproximation;
                           verbose=false,
                           optimmethod::Optim.AbstractOptimizer=BFGS(),
                           optimautodiff=:finite,
                           optimoptions::Optim.Options=Optim.Options(show_trace=verbose,                                 # Print progress
                                                                     g_tol=1e-5))
  trf = totransform(m.param)
  vfixeffs = TransformVariables.inverse(trf, fixeffs)
  o = optimize(s -> marginal_nll(m, data, TransformVariables.transform(trf, s), approx),
               vfixeffs, optimmethod, optimoptions, autodiff=optimautodiff)

  return FittedPuMaSModel(m, data, TransformVariables.transform(trf, o.minimizer), approx)
end

marginal_nll(       f::FittedPuMaSModel) = marginal_nll(       f.model, f.data, f.fixeffs, f.approx)
marginal_nll_nonmem(f::FittedPuMaSModel) = marginal_nll_nonmem(f.model, f.data, f.fixeffs, f.approx)
