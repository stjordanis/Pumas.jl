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
_lpdf(d::Distributions.Sampleable, x::Missing) = zval(d)
_lpdf(d::Distributions.UnivariateDistribution, x::AbstractVector) = sum(t -> _lpdf(d, t), x)
_lpdf(d::Distributions.MultivariateDistribution, x::AbstractVector) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable, x::PDMat) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable, x::Number) = isnan(x) ? zval(d) : logpdf(d,x)
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
    conditional_nll(m::PuMaSModel, subject::Subject, param, args...; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject` with parameters `param` and
random effects `param`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
conditional_nll(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                args...; kwargs...) =
  first(conditional_nll_ext(m, subject, param, args...; kwargs...))

function conditional_nll_ext(m::PuMaSModel,
                             subject::Subject,
                             param::NamedTuple,
                             randeffs::NamedTuple=sample_randeffs(m, param),
                             args...; kwargs...)

  # Extract a vector of the time stamps for the observations
  obstimes = subject.time
  isnothing(obstimes) && throw(ArgumentError("no observations for subject"))

  # collate that arguments
  collated = m.pre(param, randeffs, subject)

  # create solution object. By passing saveat=obstimes, we compute the solution only
  # at obstimes such that we can simply pass solution.u to m.derived
  solution = _solve(m, subject, collated, args...; saveat=obstimes, kwargs...)

  if solution === nothing
    derived_dist = m.derived(collated, solution, obstimes, subject)
  else
    # if solution contains NaN return Inf
    if (solution.retcode != :Success && solution.retcode != :Terminated) ||
        # FIXME! Make this uniform across the two solution types
        any(isnan, solution isa PKPDAnalyticalSolution ? solution(obstimes)[end] : solution.u[end])
      # FIXME! Do we need to make this type stable?
      return Inf, nothing
    end

    # extract distributions
    derived_dist = m.derived(collated, solution, obstimes, subject)
  end

  ll = _lpdf(derived_dist, subject.observations)
  return -ll, derived_dist
end

function conditional_nll(m::PuMaSModel,
                         subject::Subject,
                         param::NamedTuple,
                         randeffs::NamedTuple,
                         approx::Union{Laplace,FOCE},
                         args...; kwargs...)
  l, dist    = conditional_nll_ext(m, subject, param, randeffs, args...; kwargs...)

  # If (negative) likehood is infinity or the model is Homoscedastic, we just return l
  if isinf(l) || _is_homoscedastic(dist)
    return l
  else # compute the adjusted (negative) likelihood where the variance is evaluated at η=0
    l0, dist0 = conditional_nll_ext(m, subject, param, map(zero, randeffs), args...; kwargs...)
    return _conditional_nll(dist, dist0, subject)
  end
end

# FIXME! Having both a method for randeffs::NamedTuple and vrandeffs::AbstractVector shouldn't really be necessary. Clean it up!
function conditional_nll(m::PuMaSModel, subject::Subject, param::NamedTuple, vrandeffs::AbstractVector, approx::Union{Laplace,FOCE}, args...; kwargs...)
  rfxset = m.random(param)
  randeffs = TransformVariables.transform(totransform(rfxset), vrandeffs)
  conditional_nll(m, subject, param, randeffs, approx, args...; kwargs...)
end

function _conditional_nll(derived_dist::NamedTuple, derived_dist0::NamedTuple, subject::Subject)
  x = sum(propertynames(subject.observations)) do k
    sum(zip(getproperty(derived_dist,k),getproperty(derived_dist0,k),getproperty(subject.observations,k))) do (d,d0,obs)
      _lpdf(Normal(mean(d),sqrt(var(d0))), obs)
    end
  end
  return -x
end

"""
    penalized_conditional_nll(m::PuMaSModel, subject::Subject, param, param, args...; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as
[`conditional_nll`](@ref), except that it incorporates the penalty from the prior
distribution of the random effects.

Here `param` can be either a `NamedTuple` or a vector (representing a transformation of the
random effects to Cartesian space).
"""
function penalized_conditional_nll(m::PuMaSModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   randeffs::NamedTuple,
                                   args...;kwargs...)
  rfxset = m.random(param)
  conditional_nll(m, subject, param, randeffs, args...;kwargs...) - _lpdf(rfxset.params, randeffs)
end

function penalized_conditional_nll(m::PuMaSModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   vrandeffs::AbstractVector,
                                   args...;kwargs...)
  rfxset = m.random(param)
  randeffs = TransformVariables.transform(totransform(rfxset), vrandeffs)
  conditional_nll(m, subject, param, randeffs, args...;kwargs...) - _lpdf(rfxset.params, randeffs)
end

function penalized_conditional_nll_fn(m::PuMaSModel,
                                      subject::Subject,
                                      param::NamedTuple,
                                      args...;kwargs...)
  y -> penalized_conditional_nll(m, subject, param, y, args...; kwargs...)
end



"""
    penalized_conditional_nll!(diffres::DiffResult, m::PuMaSModel, subject::Subject, param, param, args...;
                               hessian=diffres isa HessianResult, kwargs...)

Compute the penalized conditional negative log-likelihood (see [`penalized_conditional_nll`](@ref)),
storing the gradient (and optionally, the Hessian) wrt to `param` in `diffres`.
"""
function penalized_conditional_nll!(diffres::DiffResult,
                                    m::PuMaSModel,
                                    subject::Subject,
                                    param::NamedTuple,
                                    vrandeffs::AbstractVector,
                                    args...;
                                    hessian=isa(diffres, DiffResult{2}),
                                    kwargs...)
  f = penalized_conditional_nll_fn(m, subject, param, args...; kwargs...)
  if hessian
    ForwardDiff.hessian!(diffres, f, vrandeffs)
  else
    ForwardDiff.gradient!(diffres, f, vrandeffs)
  end
end


function _initial_randeffs(m::PuMaSModel, param::NamedTuple)
  rfxset = m.random(param)
  p = TransformVariables.dimension(totransform(rfxset))

  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(param), numtype(param))
  zeros(T, p)
end

"""
randeffs_estimate(model, subject, param, approx, ...)
The point-estimate the random effects (being the mode of the empirical Bayes estimate) of a
particular subject at a particular parameter values. The result is returned as a vector
(transformed into Cartesian space).
"""
function randeffs_estimate(m::PuMaSModel,
  subject::Subject,
  param::NamedTuple,
  approx::LikelihoodApproximation,
  args...; kwargs...)
  initial_randeff = _initial_randeffs(m, param)

  return randeffs_estimate!(initial_randeff, m, subject, param, approx, args...; kwargs...)
end

function randeffs_estimate!(vrandeffs::AbstractVector,
                            m::PuMaSModel,
                            subject::Subject,
                            param::NamedTuple,
                            approx::Union{LaplaceI,FOCEI},
                            args...; kwargs...)

  cost = _η -> penalized_conditional_nll(m, subject, param, (η=_η,), args...; kwargs...)

  vrandeffs .= Optim.minimizer(Optim.optimize(
                               cost,
                               vrandeffs,
                               BFGS(linesearch=Optim.LineSearches.BackTracking(),
                                    # Make sure that step isn't too large by scaling initial Hessian by the norm of the initial gradient
                                    initial_invH=t -> Matrix(I/norm(Optim.DiffEqDiffTools.finite_difference_gradient(cost, vrandeffs)), length(vrandeffs), length(vrandeffs)));
                               autodiff=:forward))

  return vrandeffs
end

function randeffs_estimate!(vrandeffs::AbstractVector,
                            m::PuMaSModel,
                            subject::Subject,
                            fixeffs::NamedTuple,
                            approx::Union{Laplace,FOCE},
                            args...; kwargs...)

  cost = _η -> penalized_conditional_nll(m, subject, fixeffs, (η=_η,), approx, args...; kwargs...)

  vrandeffs .= Optim.minimizer(Optim.optimize(
                               cost,
                               vrandeffs,
                               BFGS(linesearch=Optim.LineSearches.BackTracking(),
                                    # Make sure that step isn't too large by scaling initial Hessian by the norm of the initial gradient
                                    initial_invH=t -> Matrix(I/norm(Optim.DiffEqDiffTools.finite_difference_gradient(cost, vrandeffs)), length(vrandeffs), length(vrandeffs)));
                                    autodiff=:forward))
  return vrandeffs
end

function randeffs_estimate!(vrandeffs::AbstractVector,
                            m::PuMaSModel,
                            subject::Subject,
                            param::NamedTuple,
                            ::Union{FO,FOI},
                            args...; kwargs...)
  vrandeffs.=zero(vrandeffs)
  vrandeffs
end

"""
    randeffs_estimate_dist(model, subject, param, approx, ...)

Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a vector (transformed into Cartesian space).
"""
function randeffs_estimate_dist(m::PuMaSModel,
                                subject::Subject,
                                param::NamedTuple,
                                approx::LaplaceI,
                                args...; kwargs...)
  vrandeffs = randeffs_estimate(m, subject, param, approx, args...; kwargs...)
  diffres = DiffResults.HessianResult(vrandeffs)
  penalized_conditional_nll!(diffres, m, subject, param, vrandeffs, args...; hessian=true, kwargs...)
  MvNormal(vrandeffs, Symmetric(inv(DiffResults.hessian(diffres))))
end

"""
    marginal_nll(model, subject, param[, param], approx, ...)
    marginal_nll(model, population, param, approx, ...)

Compute the marginal negative loglikelihood of a subject or dataset, using the integral
approximation `approx`. If no random effect (`param`) is provided, then this is estimated
from the data.

See also [`deviance`](@ref).
"""
function marginal_nll(m::PuMaSModel,
                      subject::Subject,
                      param::NamedTuple,
                      randeffs::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...; kwargs...)
  rfxset = m.random(param)
  vrandeffs = TransformVariables.inverse(totransform(rfxset), randeffs)
  marginal_nll(m, subject, param, vrandeffs, approx, args...; kwargs...)
end

function marginal_nll(m::PuMaSModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      approx::LaplaceI,
                      args...; kwargs...)
  diffres = DiffResults.HessianResult(vrandeffs)
  penalized_conditional_nll!(diffres, m, subject, param, vrandeffs, args...; hessian=true, kwargs...)
  g, m, W = DiffResults.value(diffres), DiffResults.gradient(diffres), DiffResults.hessian(diffres)
  CW = cholesky!(Symmetric(W), check=false) # W is positive-definite, only compute Cholesky once.
  if issuccess(CW)
    p = length(vrandeffs)
    g - (p*log(2π) - logdet(CW) + dot(m,CW\m))/2
  else
    return typeof(g)(Inf)
  end
end

function marginal_nll(m::PuMaSModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      ::FOCEI,
                      args...; kwargs...)

  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  _conditional_nll_ext = _η -> conditional_nll_ext(m, subject, param, (η=_η,), args...; kwargs...)

  # Construct vector of dual numbers for the random effects to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(_conditional_nll_ext, vrandeffs)
  ForwardDiff.seed!(cfg.duals, vrandeffs, cfg.seeds)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  nl_d, dist_d = _conditional_nll_ext(cfg.duals)

  nl = ForwardDiff.value(nl_d)

  if isfinite(nl)
    # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
    # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
    Ω = cov(m.random(param).params.η)
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

# Helper function to detect homoscedasticity. For now, it is assumed the input is a NamedTuple
# with a dv field containing a vector of distributions with ForwardDiff element types.
function _is_homoscedastic(dist::NamedTuple)
  # FIXME! Eventually we should support more dependent variables instead of hard coding for dv
  dv = dist.dv
  v1 = ForwardDiff.value(var(first(dv)))
  return all(t -> ForwardDiff.value(var(t)) == v1, dv)
end

function marginal_nll(m::PuMaSModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      ::FOCE,
                      args...; kwargs...)

  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  _conditional_nll_ext =  _η -> conditional_nll_ext(m, subject, param, (η=_η,), args...; kwargs...)

  # Construct vector of dual numbers for the random effects to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(_conditional_nll_ext, vrandeffs)
  ForwardDiff.seed!(cfg.duals, vrandeffs, cfg.seeds)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  nl_d, dist_d = _conditional_nll_ext(cfg.duals)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation for η=0
  # If the model is homoscedastic, it is not necessary to recompute the variances at η=0
  if _is_homoscedastic(dist_d)
    nl = ForwardDiff.value(nl_d)

    # FIXME! Don't hardcode for dv
    d_d = first(dist_d.dv)
    d_0 = Normal(ForwardDiff.value(mean(d_d)), ForwardDiff.value(std(d_d)))
    W = FIM(dist_d.dv, d_0, FOCE())
  else # in the Heteroscedastic case, compute the variances at η=0
    nl_0, dist_0 = conditional_nll_ext(m, subject, param, (η=zero(vrandeffs),), args...; kwargs...)

    # Extract the value of the conditional likelihood
    nl = ForwardDiff.value(_conditional_nll(dist_d, dist_0, subject))

    # Compute the Hessian approxmation in the random effect vector η
    # FIXME! Don't hardcode for dv
    W = FIM(dist_d.dv, dist_0.dv, FOCE())
  end

  # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  FΩ = cholesky(Ω, check=false)

  # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
  if issuccess(FΩ)
    return nl + (logdet(FΩ) + vrandeffs'*(FΩ\vrandeffs) + logdet(inv(FΩ) + W))/2
  else # Ω is numerically singular
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PuMaSModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      ::FO,
                      args...; kwargs...)

  # For FO, the conditional likelihood must be evaluated at η=0
  @assert iszero(vrandeffs)

  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  _conditional_nll_ext = _η -> conditional_nll_ext(m, subject, param, (η=_η,), args...; kwargs...)

  # Construct vector of dual numbers for the random effects to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(_conditional_nll_ext, vrandeffs)
  ForwardDiff.seed!(cfg.duals, vrandeffs, cfg.seeds)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  nl_d, dist_d = _conditional_nll_ext(cfg.duals)

  # Extract the value of the conditional likelihood
  nl = ForwardDiff.value(nl_d)

  # Compute the gradient of the likelihood and Hessian approxmation in the random effect vector η
  W, dldη = FIM(subject.observations, dist_d, FO())

  # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  FΩ = cholesky(Ω, check=false)

  # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
  if issuccess(FΩ)
    invFΩ = inv(FΩ)
    return nl + (logdet(FΩ) - dldη'*((invFΩ + W)\dldη) + logdet(invFΩ + W))/2
  else # Ω is numerically singular
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PuMaSModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      approx::Laplace,
                      args...; kwargs...)
    # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  nl = conditional_nll(m, subject, param, vrandeffs, approx, args...; kwargs...)
  W = ForwardDiff.hessian(s -> conditional_nll(m, subject, param, s, approx, args...; kwargs...), vrandeffs)
  return nl + (logdet(Ω) + vrandeffs'*(Ω\vrandeffs) + logdet(inv(Ω) + W))/2
end

function marginal_nll(m::PuMaSModel,
                      subject::Subject,
                      param::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...;
                      kwargs...)
  vrandeffs = randeffs_estimate(m, subject, param, approx, args...; kwargs...)
  marginal_nll(m, subject, param, vrandeffs, approx, args...; kwargs...)
end
function marginal_nll(m::PuMaSModel,
                      data::Population,
                      args...; kwargs...)
  sum(subject -> marginal_nll(m, subject, args...; kwargs...), data.subjects)
end


# deviance is NONMEM-equivalent marginal negative loglikelihood
"""
    deviance(model, subject, param[, param], approx, ...)
    deviance(model, data, param, approx, ...)

Compute the deviance of a subject or dataset:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
StatsBase.deviance(m::PuMaSModel,
                   subject::Subject,
                   args...; kwargs...) =
    2marginal_nll(m, subject, args...; kwargs...) - length(first(subject.observations))*log(2π)

StatsBase.deviance(m::PuMaSModel,
                   data::Population,
                   args...; kwargs...) =
    2marginal_nll(m, data, args...; kwargs...) - sum(subject->length(first(subject.observations)), data.subjects)*log(2π)
# NONMEM doesn't allow ragged, so this suffices for testing

# Compute the gradient of marginal_nll without solving inner optimization
# problem. This functions follows the approach of Almquist et al. (2015) by
# computing the gradient
#
# dL/dθ = ∂ℓ/∂θ + dη/dθ'*∂ℓ/∂η
#
# where L is the marginal likelihood of the subject, ℓ is the penalized
# conditional likelihood function and dη/dθ is the Jacobian of the optimal
# value of η with respect to the population parameters θ
function marginal_nll_gradient!(G::AbstractVector,
                                model::PuMaSModel,
                                subject::Subject,
                                param::NamedTuple,
                                randeffs::NamedTuple,
                                approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                                trf::TransformVariables.TransformTuple
                                )

  vparam = TransformVariables.inverse(trf, param)

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  ∂ℓ∂θ = Optim.DiffEqDiffTools.finite_difference_gradient(
    _param -> marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _param),
      randeffs,
      approx
    ),
    vparam
  )

  ∂ℓ∂η = Optim.DiffEqDiffTools.finite_difference_gradient(
    _η -> marginal_nll(
      model,
      subject,
      param,
      (η=_η,),
      approx
    ),
    randeffs.η
  )

  # Compute second order derivatives in high precision with ForwardDiff
  ∂²ℓ∂η² = ForwardDiff.hessian(
    t -> penalized_conditional_nll(model, subject, param, (η=t,), approx),
    randeffs.η
  )

  ∂²ℓ∂η∂θ = ForwardDiff.jacobian(
    _θ -> ForwardDiff.gradient(
      _η -> penalized_conditional_nll(model,
                                      subject,
                                      TransformVariables.transform(trf, _θ),
                                      (η=_η,),
                                      approx),
      randeffs.η
    ),
    vparam
  )

  dηdθ = -∂²ℓ∂η² \ ∂²ℓ∂η∂θ

  G .= ∂ℓ∂θ .+ dηdθ'*∂ℓ∂η

  return G
end

# Similar to the version for FOCE, FOCEI, Laplace, and LaplaceI
# but much simpler since the expansion point in η is fixed. Hence,
# the gradient is simply the partial derivative in θ
function marginal_nll_gradient!(G::AbstractVector,
                                model::PuMaSModel,
                                subject::Subject,
                                param::NamedTuple,
                                randeffs::NamedTuple,
                                approx::Union{FO,FOI},
                                trf::TransformVariables.TransformTuple
                                )

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  ∂ℓ∂θ = Optim.DiffEqDiffTools.finite_difference_gradient(
    _param -> marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _param),
      randeffs,
      approx
    ),
    TransformVariables.inverse(trf, param)
  )

  G .= ∂ℓ∂θ

  return G
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

# Homoscedastic case
function FIM(dv::AbstractVector, dv0::Distribution, ::FOCE)
  # Loop through the distribution vector and extract derivative information
  H = sum(1:length(dv)) do j
    r_inv = inv(var(dv0))
    f = SVector(ForwardDiff.partials(mean(dv[j])).values)
    f*r_inv*f'
  end

  return H
end

# Heteroscedastic case
function FIM(dv::AbstractVector, dv0::AbstractVector, ::FOCE)
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
struct FittedPuMaSModel{T1<:PuMaSModel,T2<:Population,T3,T4<:LikelihoodApproximation, T5}
  model::T1
  data::T2
  optim::T3
  approx::T4
  vvrandeffs::T5
end
const _subscriptvector = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
_to_subscript(number) = join([_subscriptvector[parse(Int32, dig)] for dig in string(number)])

function Base.show(io::IO, mime::MIME"text/plain", fpm::FittedPuMaSModel)
  level = 0.95
  quant = quantile(Normal(), level + (1.0 - level)/2)
  println(io, "FittedPuMaSModel")
  println(io)
  println(io, "Successful minimization: $(Optim.converged(fpm.optim))")
  println(io)
  println(io, "Objective function value: $(Optim.minimum(fpm.optim))")
  println(io)
  println(io, "Number of observation records: $(sum([length(sub.time) for sub in fpm.data.subjects]))")
  println(io, "Number of subjects: $(length(fpm.data.subjects))")
  println(io)
  # Get all names
  standard_errors = stderror(fpm)
  paramnames = []
  paramvals = []
  paramrse = []
  paramconfint = []
  parmastd = []
  for (paramname, paramval) in pairs(fpm.param)
    std = standard_errors[paramname]
    if typeof(paramval) <: PDMat
      std = std.mat
      mat = paramval.mat
      for j = 1:size(mat, 2)
         for i = j:size(mat, 1)
            push!(paramnames, string(paramname)*"$(_to_subscript(i)),$(_to_subscript(j))")
            push!(paramvals, string(round(mat[i,j]; sigdigits=5)))
            push!(paramrse, string(round(mat[i,j]/std[i,j]; sigdigits=5)))
            push!(paramconfint, string("[", round(mat[i,j]-std[i,j]*quant; sigdigits=5),";", round(mat[i,j]+std[i,j]*quant; sigdigits=5), "]"))
         end
      end
   elseif typeof(paramval) <: PDiagMat
      mat = paramval.diag
      std = std.diag
       for i = 1:length(mat)
          push!(paramnames, string(paramname)*"$(_to_subscript(i)),$(_to_subscript(i))")
          push!(paramvals, string(round(mat[i]; sigdigits=5)))
          push!(paramrse, string(round(mat[i]/std[i]; sigdigits=5)))
          push!(paramconfint, string("[", round(mat[i]-std[i]*quant; sigdigits=5),";", round(mat[i]+std[i]*quant; sigdigits=5), "]"))
       end
    else
      push!(paramnames, string(paramname))
      push!(paramvals, string(round(paramval; sigdigits=5)))
      push!(paramrse, string(round(paramval/std; sigdigits=5)))
      push!(paramconfint, string("[", round(paramval-std*quant; sigdigits=5),";", round(paramval+std*quant; sigdigits=5), "]"))
    end
  end

  maxname = maximum(length.(paramnames))+1
  maxval = max(maximum(length.(paramvals))+1, length("Estimate "))
  maxrs = max(maximum(length.(paramrse))+1, length("RSE "))
  maxconfint = max(maximum(length.(paramconfint))+1, length("95%-conf. int. "))
  labels = " "^(maxname+1)*rpad("Estimate ", maxval)*rpad("RSE ", maxrs)*rpad("95%-conf. int.", maxconfint)
  stringrows = []
  for (name, val, rse, confint) in zip(paramnames, paramvals, paramrse, paramconfint)
    push!(stringrows, string(rpad(name, maxname), lpad(val, maxval), lpad(rse, maxrs), lpad(confint, maxconfint)))
  end
  println(io, labels)
  for stringrow in stringrows
    println(io, stringrow)
  end
end
TreeViews.hastreeview(x::FittedPuMaSModel) = true
function TreeViews.treelabel(io::IO,x::FittedPuMaSModel,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io,mime,Base.Text(Base.summary(x)))
end


function DEFAULT_OPTIMIZE_FN(cost, p)
  Optim.optimize(cost,
                 p,
                 BFGS(linesearch=Optim.LineSearches.BackTracking(),
                      # Make sure that step isn't too large by scaling initial Hessian by the norm of the initial gradient
                      initial_invH=t -> Matrix(I/norm(Optim.DiffEqDiffTools.finite_difference_gradient(cost, p)), length(p), length(p))
                 ),
                 Optim.Options(show_trace=false, # Print progress
                               store_trace=true,
                               extended_trace=true,
                               g_tol=1e-3),
                 autodiff=:finite)
end

function Distributions.fit(m::PuMaSModel,
                           data::Population,
                           param::NamedTuple,
                           approx::LikelihoodApproximation,
                           args...;
                           optimize_fn = DEFAULT_OPTIMIZE_FN,
                           kwargs...)
  trf = totransform(m.param)
  vparam = TransformVariables.inverse(trf, param)
  cost = s -> marginal_nll(m, data, TransformVariables.transform(trf, s), approx, args...; kwargs...)
  o = optimize_fn(cost,vparam)

  # FIXME! The random effects are computed during the optimization so we should just store while optimizing instead of recomputing afterwards.
  vvrandeffs = [randeffs_estimate(m, subject, TransformVariables.transform(trf, o.minimizer), approx, args...) for subject in data.subjects]
  return FittedPuMaSModel(m, data, o, approx, vvrandeffs)
end

opt_minimizer(o::Optim.OptimizationResults) = Optim.minimizer(o)

function Base.getproperty(f::FittedPuMaSModel{<:Any,<:Any,<:Optim.MultivariateOptimizationResults}, s::Symbol)
  if s === :param
    trf = totransform(f.model.param)
    TransformVariables.transform(trf, opt_minimizer(f.optim))
  else
    return getfield(f, s)
  end
end

marginal_nll(      f::FittedPuMaSModel) = marginal_nll(f.model, f.data, f.param, f.approx)
StatsBase.deviance(f::FittedPuMaSModel) = deviance(    f.model, f.data, f.param, f.approx)

function _observed_information(f::FittedPuMaSModel, ::Val{Score}) where Score
  # Transformation the NamedTuple of parameters to a Vector
  # without applying any bounds (identity transform)
  trf = toidentitytransform(f.model.param)
  vparam = TransformVariables.inverse(trf, f.param)

  # Initialize arrays
  H = zeros(eltype(vparam), length(vparam), length(vparam))
  if Score
    S = copy(H)
    g = similar(vparam, length(vparam))
  else
    S = g = nothing
  end

  # Loop through subject and compute Hessian and score contributions
  for i in 1:length(f.data)
    subject = f.data[i]

    # Compute Hessian contribution and update Hessian
    H .+= Optim.DiffEqDiffTools.finite_difference_jacobian(vparam) do _j, _param
      param = TransformVariables.transform(trf, _param)
      vrandeffs = randeffs_estimate(f.model, subject, param, f.approx)
      marginal_nll_gradient!(_j, f.model, subject, param, (η=vrandeffs,), f.approx, trf)
      return nothing
    end

    if Score
      # Compute score contribution
      vrandeffs = randeffs_estimate(f.model, subject, f. param, f.approx)
      marginal_nll_gradient!(g, f.model, subject, f.param, (η=vrandeffs,), f.approx, trf)

      # Update outer product of scores
      S .+= g .* g'
    end
  end
  return H, S
end

function _expected_information(m::PuMaSModel,
                               subject::Subject,
                               param::NamedTuple,
                               randeffs::NamedTuple,
                               ::FO,
                               args...; kwargs...)

  trf = toidentitytransform(m.param)
  vparam = TransformVariables.inverse(trf, param)

  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  __E_and_V = _param -> _E_and_V(m, subject, TransformVariables.transform(trf, _param), randeffs.η, FO(), args...; kwargs...)

  # Construct vector of dual numbers for the population parameters to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(__E_and_V, vparam)
  ForwardDiff.seed!(cfg.duals, vparam, cfg.seeds)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  E_d, V_d = __E_and_V(cfg.duals)

  V⁻¹ = inv(cholesky(ForwardDiff.value.(V_d)))
  dEdθ = hcat((collect(ForwardDiff.partials(E_k).values) for E_k in E_d)...)

  m = size(dEdθ, 1)
  n = size(dEdθ, 2)
  dVpart = similar(dEdθ, m, m)
  for l in 1:m
    dVdθl = [ForwardDiff.partials(V_d[i,j]).values[l] for i in 1:n, j in 1:n]
    for k in 1:m
      dVdθk = [ForwardDiff.partials(V_d[i,j]).values[k] for i in 1:n, j in 1:n]
      # dVpart[l,k] = tr(dVdθk * V⁻¹ * dVdθl * V⁻¹)/2
      dVpart[l,k] = sum((V⁻¹ * dVdθk) .* (dVdθl * V⁻¹))/2
    end
  end

  return dEdθ*V⁻¹*dEdθ' + dVpart
end

function StatsBase.informationmatrix(f::FittedPuMaSModel; expected::Bool=true)
  data     = f.data
  model    = f.model
  param    = f.param
  vrandeffs = f.vrandeffs
  if expected
    return sum(_expected_information(model, data[i], param, (η=vrandeffs[i],), f.approx) for i in 1:length(data))
  else
    return first(_observed_information(f, Val(false)))
  end
end

"""
    vcov(f::FittedPuMaSModel) -> Matrix

Compute the covariance matrix of the population parameters
"""
function StatsBase.vcov(f::FittedPuMaSModel)

  # Compute the observed information based on the Hessian (H) and the product of the outer scores (S)
  H, S = _observed_information(f)

  # Use generialized eigenvalue decomposition to compute inv(H)*S*inv(H)
  F = eigen(Symmetric(H), Symmetric(S))
  any(t -> t <= 0, F.values) && @warn("Hessian is not positive definite")
  return F.vectors*Diagonal(inv.(abs2.(F.values)))*F.vectors'
end

"""
    stderror(f::FittedPuMaSModel) -> NamedTuple

Compute the standard errors of the population parameters and return
the result as a `NamedTuple` matching the `NamedTuple` of population
parameters.
"""
function StatsBase.stderror(f::FittedPuMaSModel)
  trf = toidentitytransform(f.model.param)
  ss = sqrt.(diag(vcov(f)))
  return TransformVariables.transform(trf, ss)
end

function _E_and_V(m::PuMaSModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector,
                  ::FO,
                  args...; kwargs...)

  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m, subject, param, (η=vrandeffs,))
  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return mean.(dist.dv), V
end

function _E_and_V(m::PuMaSModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector,
                  ::FOCE,
                  args...; kwargs...)

  y = subject.observations.dv
  nl0, dist0 = conditional_nll_ext(m, subject, param, (η=zero(vrandeffs),))
  nl , dist  = conditional_nll_ext(m, subject, param, (η=vrandeffs,))

  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist0.dv)))

  return mean.(dist.dv) .- F*vrandeffs, V

end

# Some type piracy for the time being
Distributions.MvNormal(D::Diagonal) = MvNormal(PDiagMat(D.diag))
