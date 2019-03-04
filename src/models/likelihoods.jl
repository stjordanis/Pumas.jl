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
_lpdf(d::Distributions.Sampleable,x) = x === missing ? zval(d) : logpdf(d,x)
_lpdf(d::Distributions.Sampleable,x::Number) = isnan(x) ? zval(d) : logpdf(d,x)
_lpdf(d::Constrained, x) = _lpdf(d.dist, x)
function _lpdf(ds::T, xs::S) where {T<:NamedTuple, S<:NamedTuple}
  sum(keys(xs)) do k
    haskey(ds, k) ? _lpdf(getproperty(ds, k), getproperty(xs, k)) : zero(getproperty(xs, k))
  end
end
_lpdf(d::Domain, x) = 0.0
# Define two vectorized helper functions __lpdf. Once we can do two argument mapreduce
# or the equivalent without allocations, it should be possible to get rid of these.
function __lpdf(ds::Distributions.Sampleable, xs::AbstractVector)
  l = _lpdf(ds, xs[1])
  @inbounds for i in 2:length(ds)
    l += _lpdf(ds, xs[i])
  end
  return l
end
function __lpdf(ds::AbstractVector, xs::AbstractVector)
  if length(ds) != length(xs)
    throw(DimensionMismatch("vectors must have same length"))
  end
  l = _lpdf(ds[1], xs[1])
  @inbounds for i in 2:length(ds)
    l += _lpdf(ds[i], xs[i])
  end
  return l
end
function _lpdf(derived_dist::NamedTuple, observations::StructArray)
  obs = StructArrays.fieldarrays(observations)
  sum(keys(obs)) do k
    if haskey(derived_dist, k)
      __lpdf(getfield(derived_dist, k), getfield(obs, k))
    end
  end
end

"""
    conditional_nll(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
conditional_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, args...; kwargs...) =
  first(conditional_nll_ext(m, subject, x0, args...; kwargs...))

function conditional_nll_ext(m::PKPDModel, subject::Subject, x0::NamedTuple,
                             y0::NamedTuple=rand_random(m, x0), args...; kwargs...)
  # Extract a vector of the time stamps for the observations
  obstimes = subject.time
  isempty(obstimes) && throw(ArgumentError("no observations for subject"))

  # collate that arguments
  collated = m.pre(x0, y0, subject.covariates)

  # create solution object
  solution = _solve(m, subject, collated, args...; kwargs...)

  if solution === nothing
     vals, derived_dist = m.derived(collated, nothing, obstimes)
  else
    # compute solution values
    path = solution.(obstimes, continuity=:right)

    # if solution contains NaN return Inf
    if any(isnan, last(path)) || solution.retcode != :Success
      # FIXME! Do we need to make this type stable?
      return Inf, nothing, nothing
    end

    # extract values and distributions
    vals, derived_dist = m.derived(collated, path, obstimes) # the second component is distributions
  end

  ll = _lpdf(derived_dist, subject.observations)
  return -ll, vals, derived_dist
end

function conditional_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, approx::Union{Laplace,FOCE}, args...; kwargs...)
  l, vals, dist    = conditional_nll_ext(m, subject, x0, y0, args...; kwargs...)
  if isinf(l)
    return l
  end
  l0, vals0, dist0 = conditional_nll_ext(m, subject, x0, map(zero, y0), args...; kwargs...)
  conditional_nll(dist, dist0, subject)
end

# FIXME! Having both a method for y0::NamedTuple and vy0::AbstractVector shouldn't really be necessary. Clean it up!
function conditional_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::Union{Laplace,FOCE}, args...; kwargs...)
  rfxset = m.random(x0)
  y0 = TransformVariables.transform(totransform(rfxset), vy0)
  conditional_nll(m, subject, x0, y0, approx, args...; kwargs...)
end

function conditional_nll(derived_dist, derived_dist0, subject)
  typ = flattentype(derived_dist)
  n = length(derived_dist)
  x = sum(enumerate(subject.observations)) do (idx,obs)
    if eltype(derived_dist) <: Array
      _lpdf(typ(ntuple(i->Normal(mean(derived_dist[i][idx]),sqrt(var(derived_dist0[i][idx]))), n)), obs)
    else
      _lpdf(Normal(mean(derived_dist),sqrt(var(derived_dist0))), obs)
    end
  end
  return -x
end

"""
    penalized_conditional_nll(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as
[`conditional_nll`](@ref), except that it incorporates the penalty from the prior
distribution of the random effects.

Here `rfx` can be either a `NamedTuple` or a vector (representing a transformation of the
random effects to Cartesian space).
"""
function penalized_conditional_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, args...;kwargs...)
  rfxset = m.random(x0)
  conditional_nll(m, subject, x0, y0, args...;kwargs...) - _lpdf(rfxset.params, y0)
end

function penalized_conditional_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, args...;kwargs...)
  rfxset = m.random(x0)
  y0 = TransformVariables.transform(totransform(rfxset), vy0)
  conditional_nll(m, subject, x0, y0, args...;kwargs...) - _lpdf(rfxset.params, y0)
end

function penalized_conditional_nll_fn(m::PKPDModel, subject::Subject, x0::NamedTuple, args...;kwargs...)
  y -> penalized_conditional_nll(m, subject, x0, y, args...; kwargs...)
end



"""
    penalized_conditional_nll!(diffres::DiffResult, m::PKPDModel, subject::Subject, param, rfx, args...;
                               hessian=diffres isa HessianResult, kwargs...)

Compute the penalized conditional negative log-likelihood (see [`penalized_conditional_nll`](@ref)),
storing the gradient (and optionally, the Hessian) wrt to `rfx` in `diffres`.
"""
function penalized_conditional_nll!(diffres::DiffResult, m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, args...;
                                    hessian=isa(diffres, DiffResult{2}), kwargs...)
  f = penalized_conditional_nll_fn(m, subject, x0, args...; kwargs...)
  if hessian
    ForwardDiff.hessian!(diffres, f, vy0)
  else
    ForwardDiff.gradient!(diffres, f, vy0)
  end
end


"""
    rfx_estimate(model, subject, param, approx, ...)

The point-estimate the random effects (being the mode of the empirical Bayes estimate) of a
particular subject at a particular parameter values. The result is returned as a vector
(transformed into Cartesian space).
"""
rfx_estimate

function rfx_estimate(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::Union{LaplaceI,FOCEI}, args...; kwargs...)
  rfxset = m.random(x0)
  p = TransformVariables.dimension(totransform(rfxset))
  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(x0), numtype(x0))
  Optim.minimizer(Optim.optimize(penalized_conditional_nll_fn(m, subject, x0, args...; kwargs...), zeros(T, p), BFGS(); autodiff=:forward))
end

function rfx_estimate(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::Union{Laplace,FOCE}, args...; kwargs...)
  rfxset = m.random(x0)
  p = TransformVariables.dimension(totransform(rfxset))
  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(x0), numtype(x0))
  Optim.minimizer(Optim.optimize(s -> penalized_conditional_nll(m, subject, x0, (η=s,), approx, args...; kwargs...), zeros(T, p), BFGS(); autodiff=:forward))
end

function rfx_estimate(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::Union{FO,FOI}, args...; kwargs...)
  rfxset = m.random(x0)
  p = TransformVariables.dimension(totransform(rfxset))
  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(x0), numtype(x0))
  return zeros(T, p)
end

"""
    rfx_estimate_dist(model, subject, param, approx, ...)

Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a vector (transformed into Cartesian space).
"""
function rfx_estimate_dist(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::LaplaceI, args...; kwargs...)
  vy0 = rfx_estimate(m, subject, x0, approx, args...; kwargs...)
  diffres = DiffResults.HessianResult(vy0)
  penalized_conditional_nll!(diffres, m, subject, x0, vy0, args...; hessian=true, kwargs...)
  MvNormal(vy0, Symmetric(inv(DiffResults.hessian(diffres))))
end

"""
    marginal_nll(model, subject, param[, rfx], approx, ...)
    marginal_nll(model, population, param, approx, ...)

Compute the marginal negative loglikelihood of a subject or dataset, using the integral
approximation `approx`. If no random effect (`rfx`) is provided, then this is estimated
from the data.

See also [`marginal_nll_nonmem`](@ref).
"""
function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, approx::LikelihoodApproximation, args...;
                      kwargs...)
  rfxset = m.random(x0)
  vy0 = TransformVariables.inverse(totransform(rfxset), y0)
  marginal_nll(m, subject, x0, vy0, approx, args...; kwargs...)
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::LaplaceI, args...;
                      kwargs...)
  diffres = DiffResults.HessianResult(vy0)
  penalized_conditional_nll!(diffres, m, subject, x0, vy0, args...; hessian=true, kwargs...)
  g, m, W = DiffResults.value(diffres),DiffResults.gradient(diffres),DiffResults.hessian(diffres)
  CW = cholesky!(Symmetric(W)) # W is positive-definite, only compute Cholesky once.
  p = length(vy0)
  g - (p*log(2π) - logdet(CW) + dot(m,CW\m))/2
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::FOCEI, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  l,val,dist = conditional_nll_ext(m, subject, x0, (η=vy0,), args...; kwargs...)
  w = FIM(m, subject, x0, vy0, approx, dist, args...; kwargs...)
  l += (logdet(Ω) + vy0'*(Ω\vy0) + logdet(inv(Ω) + w))/2
  return l
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::FOCE, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  nl = conditional_nll(m, subject, x0, vy0, approx, args...; kwargs...)
  W = FIM(m,subject, x0, vy0, approx, args...; kwargs...)

  FΩ = cholesky(Ω, check=false)
  if issuccess((FΩ))
    return nl + (logdet(FΩ) + vy0'*(FΩ\vy0) + logdet(inv(FΩ) + W))/2
  else # Ω is numerically singular
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::FO, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  l0, vals0, dist0 = conditional_nll_ext(m, subject, x0, (η=zero(vy0),), args...; kwargs...)
  W, dldη = FIM(m, subject, x0, zero(vy0), approx, dist0, args...; kwargs...)

  FΩ = cholesky(Ω, check=false)
  if issuccess((FΩ))
    invFΩ = inv(FΩ)
    return l0 + (logdet(FΩ) - dldη'*((invFΩ + W)\dldη) + logdet(invFΩ + W))/2
  else # Ω is numerically singular
    return typeof(l0)(Inf)
  end
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::Laplace, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  nl = conditional_nll(m, subject, x0, vy0, approx, args...; kwargs...)
  W = ForwardDiff.hessian(s -> conditional_nll(m, subject, x0, s, approx, args...; kwargs...), vy0)
  return nl + (logdet(Ω) + vy0'*(Ω\vy0) + logdet(inv(Ω) + W))/2
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::LikelihoodApproximation, args...;
                      kwargs...)
  vy0 = rfx_estimate(m, subject, x0, approx, args...; kwargs...)
  marginal_nll(m, subject, x0, vy0, approx, args...; kwargs...)
end
function marginal_nll(m::PKPDModel, data::Population, args...;
                      kwargs...)
  sum(subject -> marginal_nll(m, subject, args...; kwargs...), data.subjects)
end


"""
    marginal_nll_nonmem(model, subject, param[, rfx], approx, ...)
    marginal_nll_nonmem(model, data, param, approx, ...)

Compute the NONMEM-equivalent marginal negative loglikelihood of a subject or dataset:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
marginal_nll_nonmem(m::PKPDModel, subject::Subject, args...; kwargs...) =
    2marginal_nll(m, subject, args...; kwargs...) - length(subject.observations)*log(2π)
marginal_nll_nonmem(m::PKPDModel, data::Population, args...; kwargs...) =
    2marginal_nll(m, data, args...; kwargs...) - sum(subject->length(subject.observations), data.subjects)*log(2π)



"""
In named tuple nt, replace the value x.var by y
"""
@generated function Base.setindex(x::NamedTuple,y,v::Val)
  k = first(v.parameters)
  k ∉ x.names ? :x : :( (x..., $k=y) )
end

function generate_enclosed_likelihood(ll, model::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, v, args...; kwargs...)
  function (z)
    _x0 = setindex(x0,z,v)
    _y0 = setindex(y0,z,v)
    ll(model, subject, _x0, _y0, args...; kwargs...)
  end
end

function ll_derivatives(ll, model::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, var::Symbol, args...; kwargs...)
  ll_derivatives(ll, model, subject::Subject, x0::NamedTuple, y0::NamedTuple, Val(var), args...; kwargs...)
end

@generated function ll_derivatives(ll, model::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple,
                                   v::Val{var}, args...;
                                   hessian_required = true,
                                   transform=false, kwargs...) where var
  var ∉ fieldnames(x0) ∪ fieldnames(y0) && error("Variable name not recognized")
  valex = var ∈ fieldnames(x0) ? :(x0.$var) : :(y0.$var)
  quote
    f = generate_enclosed_likelihood(ll,model,subject,x0,y0,v, args...; kwargs...)
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

function _mean(model, subject, x0::NamedTuple, y0::NamedTuple, i, args...; kwargs...)
  x_, vals_, dist_ = conditional_nll_ext(model, subject, x0, y0, args...; kwargs...)
  mean(dist_[1][i])
end

function _var(model, subject, x0::NamedTuple, y0::NamedTuple, i, args...; kwargs...)
  x_, vals_, dist_ = conditional_nll_ext(model, subject, x0, y0, args...; kwargs...)
  var(dist_[1][i])
end

function _mean0(model, subject, x0::NamedTuple, y0::NamedTuple, i, args...; kwargs...)
  _mean(model, subject, x0, map(zero, y0), i, args...; kwargs...)
end

function FIM(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::FOCEI, dist, args...; kwargs...)
  f_res     = DiffResults.GradientResult(vy0)
  del_r_res = DiffResults.GradientResult(vy0)
  fim = sum(1:length(subject.observations)) do j
    r_inv = inv(var(dist[1][j]))
    # FIXME! Wrapping vy0 shouldn't be necessary but currently the names are used inside ll_derivatives
    ForwardDiff.gradient!(f_res, s -> _mean(m, subject, x0, (η=s,), j, args...; kwargs...), vy0)
    f = DiffResults.gradient(f_res)
    # FIXME! Wrapping vy0 shouldn't be necessary but currently the names are used inside ll_derivatives
    ForwardDiff.gradient!(del_r_res, s -> _var(m, subject, x0, (η=s,), j, args...; kwargs...), vy0)
    del_r = DiffResults.gradient(del_r_res)
    return f*r_inv*f' + (r_inv*del_r*r_inv*del_r')/2
  end
  fim
end

function FIM(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::FOCE, args...; kwargs...)
  l0, vals0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),), args...; kwargs...)
  fim = sum(1:length(subject.observations)) do j
    r_inv = inv(var(dist0[1][j]))
    # FIXME! Wrapping vy0 shouldn't be necessary but currently the names are used inside ll_derivatives
    res = ll_derivatives(_mean, m, subject, x0, (η=vy0,), :η, j, args...; kwargs...)
    f = DiffResults.gradient(res)
    f*r_inv*f'
  end
  fim
end

function FIM(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::FO, dist0, args...; kwargs...)
  r = var(dist0[1][1])
  f = ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), 1, args...; kwargs...), vy0)
  fdr = f/r
  H = fdr*f'
  mean0 = _mean0(m, subject, x0, (η=vy0,), 1, args...; kwargs...)
  # FIXME! This is not correct when there are more than a single dependent variable
  dldη = fdr*(subject.observations[1][1] - mean0)

  for j in 2:length(subject.observations)
    r = var(dist0[1][j])
    ForwardDiff.gradient!(f, s -> _mean(m, subject, x0, (η=s,), j, args...; kwargs...), vy0)
    fdr = f/r
    H += fdr*f'
    mean0 = _mean0(m, subject, x0, (η=vy0,), j, args...; kwargs...)
    # FIXME! This is not correct when there are more than a single dependent variable
    dldη += fdr*(subject.observations[j][1] - mean0)
  end

  return H, dldη
end

# Fitting methods
struct FittedPKPDModel{T1<:PKPDModel,T2<:Population,T3<:NamedTuple,T4<:LikelihoodApproximation}
  model::T1
  data::T2
  x0::T3
  approx::T4
end

function Distributions.fit(m::PKPDModel,
                           data::Population,
                           x0::NamedTuple,
                           approx::LikelihoodApproximation;
                           verbose=false,
                           optimmethod::Optim.AbstractOptimizer=BFGS(),
                           optimautodiff=:finite,
                           optimoptions::Optim.Options=Optim.Options(show_trace=verbose,                                 # Print progress
                                                                     g_tol=1e-5))
  trf = totransform(m.param)
  vx0 = TransformVariables.inverse(trf, x0)
  o = optimize(s -> marginal_nll(m, data, TransformVariables.transform(trf, s), approx),
               vx0, optimmethod, optimoptions, autodiff=optimautodiff)

  return FittedPKPDModel(m, data, TransformVariables.transform(trf, o.minimizer), approx)
end

marginal_nll(       f::FittedPKPDModel) = marginal_nll(       f.model, f.data, f.x0, f.approx)
marginal_nll_nonmem(f::FittedPKPDModel) = marginal_nll_nonmem(f.model, f.data, f.x0, f.approx)

function npde(m::PKPDModel,subject::Subject, x0::NamedTuple,nsim)
  yi = [obs.val.dv for obs in subject.observations]
  sims = []
  for i in 1:nsim
    vals = simobs(m, subject, x0)
    push!(sims, vals.derived.dv)
  end
  mean_yi = [mean(sims[:][i]) for i in 1:length(sims[1])]
  covm_yi = cov(sims)
  covm_yi = sqrt(inv(covm_yi))
  yi_decorr = (covm_yi)*(yi .- mean_yi)
  phi = []
  for i in 1:nsim
    yi_i = sims[i]
    yi_decorr_i = (covm_yi)*(yi_i .- mean_yi)
    push!(phi,[yi_decorr_i[j]>=yi_decorr[j] ? 0 : 1 for j in 1:length(yi_decorr_i)])
  end
  phi = sum(phi)/nsim
  [quantile(Normal(),phi[i]) for i in 1:length(subject.observations)]
end

function wres(m::PKPDModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FO()))
  yi = [obs.val.dv for obs in subject.observations]
  l0, vals0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  mean_yi = (mean.(dist0[1]))
  Ω = cov(m.random(x0).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations)]
  var_yi = sqrt(inv((Diagonal(var.(dist0[1]))) + (f*Ω*f')))
  (var_yi)*(yi .- mean_yi)  
end

function cwres(m::PKPDModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCE()))
  yi = [obs.val.dv for obs in subject.observations]
  l0, vals0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  l, vals, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  Ω = cov(m.random(x0).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations)]
  var_yi = sqrt(inv((Diagonal(var.(dist0[1]))) + (f*Ω*f')))
  mean_yi = (mean.(dist[1])) .- vec(f*vy0')
  (var_yi)*(yi .- mean_yi)  
end

function cwresi(m::PKPDModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCEI()))
  yi = [obs.val.dv for obs in subject.observations]
  l, vals, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  Ω = cov(m.random(x0).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations)]
  var_yi = sqrt(inv((Diagonal(var.(dist[1]))) + (f*Ω*f')))
  mean_yi = (mean.(dist[1])) .- vec(f*vy0')
  (var_yi)*(yi .- mean_yi)  
end

function pred(m::PKPDModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FO()))
  l0, vals0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  mean_yi = (mean.(dist0[1]))
  mean_yi
end

function cpred(m::PKPDModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCE()))
  l, vals, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations)]
  mean_yi = (mean.(dist[1])) .- vec(f*vy0')
  mean_yi
end

function cpredi(m::PKPDModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCEI()))
  l, vals, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations)]
  mean_yi = (mean.(dist[1])) .- vec(f*vy0')
  mean_yi
end