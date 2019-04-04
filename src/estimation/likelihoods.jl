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
  if isinf(l) || _is_homoscedast(dist)
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
  rfxset = m.random(param)
  p = TransformVariables.dimension(totransform(rfxset))

  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(param), numtype(param))

  return randeffs_estimate(m, subject, param, zeros(T, p), approx, args...; kwargs...)
end

function randeffs_estimate(m::PuMaSModel,
                           subject::Subject,
                           param::NamedTuple,
                           vrandeffs::AbstractVector,
                           approx::Union{LaplaceI,FOCEI},
                           args...; kwargs...)
  Optim.minimizer(
    Optim.optimize(
      penalized_conditional_nll_fn(m, subject, param, args...; kwargs...),
      vrandeffs,
      BFGS(linesearch=Optim.LineSearches.BackTracking());
      autodiff=:forward
    )
  )
end

function randeffs_estimate(m::PuMaSModel,
                           subject::Subject,
                           param::NamedTuple,
                           vrandeffs::AbstractVector,
                           approx::Union{Laplace,FOCE},
                           args...; kwargs...)
  Optim.minimizer(
    Optim.optimize(
      s -> penalized_conditional_nll(m, subject, param, (η=s,), approx, args...; kwargs...),
      vrandeffs,
      BFGS(linesearch=Optim.LineSearches.BackTracking());
      autodiff=:forward
    )
  )
end

randeffs_estimate(m::PuMaSModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector,
                  ::Union{FO,FOI},
                  args...; kwargs...) = zero(vrandeffs)

"""
    randeffs_estimate_dist(model, subject, param, approx, ...)

Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a vector (transformed into Cartesian space).
"""
function randeffs_estimate_dist(m::PuMaSModel, subject::Subject, param::NamedTuple, approx::LaplaceI, args...; kwargs...)
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

See also [`marginal_nll_nonmem`](@ref).
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

function marginal_nll(m::PuMaSModel, subject::Subject, param::NamedTuple, vrandeffs::AbstractVector, approx::LaplaceI, args...;
                      kwargs...)
  diffres = DiffResults.HessianResult(vrandeffs)
  penalized_conditional_nll!(diffres, m, subject, param, vrandeffs, args...; hessian=true, kwargs...)
  g, m, W = DiffResults.value(diffres),DiffResults.gradient(diffres),DiffResults.hessian(diffres)
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
function _is_homoscedast(dist::NamedTuple)
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
  if _is_homoscedast(dist_d)
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


"""
    marginal_nll_nonmem(model, subject, param[, param], approx, ...)
    marginal_nll_nonmem(model, data, param, approx, ...)

Compute the NONMEM-equivalent marginal negative loglikelihood of a subject or dataset:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
marginal_nll_nonmem(m::PuMaSModel,
                    subject::Subject,
                    args...; kwargs...) =
    2marginal_nll(m, subject, args...; kwargs...) - length(first(subject.observations))*log(2π)

marginal_nll_nonmem(m::PuMaSModel,
                    data::Population,
                    args...; kwargs...) =
    2marginal_nll(m, data, args...; kwargs...) - sum(subject->length(first(subject.observations)), data.subjects)*log(2π)
# NONMEM doesn't allow ragged, so this suffices for testing


"""
In named tuple nt, replace the value x.var by y
"""
@generated function Base.setindex(x::NamedTuple,y,v::Val)
  k = first(v.parameters)
  k ∉ x.names ? :x : :( (x..., $k=y) )
end

function _mean(model, subject, param::NamedTuple, randeffs::NamedTuple, i, args...; kwargs...)
  x_, dist_ = conditional_nll_ext(model, subject, param, randeffs, args...; kwargs...)
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
struct FittedPuMaSModel{T1<:PuMaSModel,T2<:Population,T3,T4<:LikelihoodApproximation}
  model::T1
  data::T2
  optim::T3
  approx::T4
end

function Distributions.fit(m::PuMaSModel,
                           data::Population,
                           param::NamedTuple,
                           approx::LikelihoodApproximation,
                           args...;
                           verbose=false,
                           optimmethod::Optim.AbstractOptimizer=BFGS(linesearch=Optim.LineSearches.BackTracking()),
                           optimautodiff=:finite,
                           optimoptions::Optim.Options=Optim.Options(show_trace=verbose, # Print progress
                                                                     g_tol=1e-3),
                           kwargs...)
  trf = totransform(m.param)
  vparam = TransformVariables.inverse(trf, param)
  o = optimize(s -> marginal_nll(m, data, TransformVariables.transform(trf, s), approx, args...; kwargs...),
               vparam, optimmethod, optimoptions, autodiff=optimautodiff)

  return FittedPuMaSModel(m, data, o, approx)
end

function Base.getproperty(f::FittedPuMaSModel{<:Any,<:Any,<:Optim.MultivariateOptimizationResults}, s::Symbol)
  if s === :param
    trf = totransform(f.model.param)
    TransformVariables.transform(trf, f.optim.minimizer)
  else
    return getfield(f, s)
  end
end

marginal_nll(       f::FittedPuMaSModel) = marginal_nll(       f.model, f.data, f.param, f.approx)
marginal_nll_nonmem(f::FittedPuMaSModel) = marginal_nll_nonmem(f.model, f.data, f.param, f.approx)
