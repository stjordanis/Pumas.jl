import DiffResults: DiffResult

Base.@pure flattentype(t) = NamedTuple{fieldnames(typeof(t)), NTuple{length(t), eltype(eltype(t))}}

"""
    _lpdf(d,x)

The log pdf: this differs from `Distributions.logdpf` definintion in a couple of ways:
- if `d` is a non-distribution it assumes the Dirac distribution.
- if `x` is `NaN` or `Missing`, it returns 0.
- if `d` is a `NamedTuple` of distributions, and `x` is a `NamedTuple` of observations, it computes the sum of the observed variables.
"""
_lpdf(d::Number, x::Number) = d == x ? 0.0 : -Inf
_lpdf(d::Distributions.Sampleable,x) = x === missing ? zval(d) : logpdf(d,x)
_lpdf(d::Distributions.Sampleable,x::Number) = isnan(x) ? zval(d) : logpdf(d,x)
function _lpdf(ds::T, xs::S) where {T<:NamedTuple, S<:NamedTuple}
  sum(keys(xs)) do k
    haskey(ds, k) ? _lpdf(getproperty(ds, k), getproperty(xs, k)) : zero(getproperty(xs, k))
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

function conditional_nll_ext(m::PKPDModel, subject::Subject, x0::NamedTuple, args...; kwargs...)
  obstimes = [obs.time for obs in subject.observations]
  isempty(obstimes) && throw(ArgumentError("no observations for subject"))
  derived = derivedfun(m,subject,x0::NamedTuple, args...;kwargs...)
  vals, derived_dist = derived(obstimes) # the second component is distributions
  x = let derived_dist=derived_dist
    sum(enumerate(subject.observations)) do (idx,obs)
      if eltype(derived_dist) <: Array
        _lpdf(NamedTuple{Base._nt_names(obs.val)}(ntuple(i->derived_dist[i][idx], length(derived_dist))), obs.val)
      else
        _lpdf(derived_dist, obs.val)
      end
    end
  end
  return -x, vals, derived_dist
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
  conditional_nll(m,subject, x0, y0, args...;kwargs...) - _lpdf(rfxset.params, y0)
end

function penalized_conditional_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, args...;kwargs...)
  rfxset = m.random(x0)
  y0 = TransformVariables.transform(totransform(rfxset), vy0)
  conditional_nll(m,subject, x0, y0, args...;kwargs...) - _lpdf(rfxset.params, y0)
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


abstract type LikelihoodApproximation end
struct Laplace <: LikelihoodApproximation end
struct FOCEI <: LikelihoodApproximation end
struct FOCE <: LikelihoodApproximation end
struct FOI <: LikelihoodApproximation end
struct FO <: LikelihoodApproximation end

function conditional_nll(derived_dist, derived_dist0,subject)
  typ = flattentype(derived_dist)
  n = length(derived_dist)
  x = sum(enumerate(subject.observations)) do (idx,obs)
    if eltype(derived_dist) <: Array
      _lpdf(typ(ntuple(i->Normal(mean(derived_dist[i][idx]),sqrt(var(derived_dist0[i][idx]))), n)), obs.val)
    else
      _lpdf(Normal(mean(derived_dist),sqrt(var(derived_dist0))), obs.val)
    end
  end
  return -x
end

"""
    rfx_estimate(model, subject, param, approx, ...)

The point-estimate the random effects (being the mode of the empirical Bayes estimate) of a
particular subject at a particular parameter values. The result is returned as a vector
(transformed into Cartesian space).
"""
function rfx_estimate(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::Laplace, args...; kwargs...)
  rfxset = m.random(x0)
  p = TransformVariables.dimension(totransform(rfxset))
  Optim.minimizer(Optim.optimize(penalized_conditional_nll_fn(m, subject, x0, args...; kwargs...), zeros(p), BFGS(); autodiff=:forward))
end

"""
    rfx_estimate_dist(model, subject, param, approx, ...)

Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a vector (transformed into Cartesian space).
"""
function rfx_estimate_dist(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::Laplace, args...; kwargs...)
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

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::Laplace, args...;
                      kwargs...)
  diffres = DiffResults.HessianResult(vy0)
  penalized_conditional_nll!(diffres, m, subject, x0, vy0, args...; hessian=true, kwargs...)
  g, m, W = DiffResults.value(diffres),DiffResults.gradient(diffres),DiffResults.hessian(diffres)
  CW = cholesky!(Symmetric(W)) # W is positive-definite, only compute Cholesky once.
  p = length(vy0)
  g - (p*log(2π) - logdet(CW) + dot(m,CW\m))/2
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, approx::FOCEI, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  l,val,dist = conditional_nll_ext(m,subject,x0, y0,args...;kwargs...)
  w = FIM(m,subject, x0, y0, approx, dist, args...;kwargs...)
  return l + (logdet(Ω) + y0.η'*(Ω\y0.η) + logdet(inv(Ω) + w))/2
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, approx::FOI, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  y_ = (η = zeros(length(y0.η)),)
  l,val,dist = conditional_nll(m,subject,x0, y0,args...;extended_return = true,kwargs...)
  w = FIM(m,subject, x0, y_, FOCEI(), dist, args...;kwargs...)
  return l + (logdet(Ω) + y0.η'*(Ω\y0.η) + logdet(inv(Ω) + w))/2
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, approx::FOCE, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  l, vals, dist = conditional_nll_ext(m,subject,x0, y0, args...;kwargs...)
  l0, vals0, dist0 = conditional_nll_ext(m,subject,x0, zeros(length(y0)), args...;kwargs...)
  l_ = -conditional_nll(dist, dist0,subject) - (length(subject.observations)-2)*log(2π)/2
  w = FIM(m,subject, x0, y0, approx, dist0, args...;kwargs...)
  return -l_ + (logdet(Ω) + y0.η'*(Ω\y0.η) + logdet(inv(Ω) + w))/2
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, y0::NamedTuple, approx::FO, args...; kwargs...)
  Ω = cov(m.random(x0).params.η)
  l, vals, dist = conditional_nll(m,subject,x0, y0, args...;extended_return = true,kwargs...)
  l0, vals0, dist0 = conditional_nll(m,subject,x0, zeros(length(y0)), args...;extended_return = true,kwargs...)
  l_ = -conditional_nll(dist, dist0,subject) - (length(subject.observations)-2)*log(2π)/2
  y_ = (η = zeros(length(y0.η)),)
  w = FIM(m,subject, x0, y_, FOCE(), dist0, args...;kwargs...)
  return -l_ + (logdet(Ω) + y0.η'*(Ω\y0.η) + logdet(inv(Ω) + w))/2 
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

function generate_enclosed_likelihood(ll,model,subject,x0,y0,v, args...; kwargs...)
  function (z)
    _x0 = setindex(x0,z,v)
    _y0 = setindex(y0,z,v)
    ll(model, subject, _x0, _y0, args...; kwargs...)
  end
end

function ll_derivatives(ll,model,subject,x0,y0,var::Symbol,args...; kwargs...)
  ll_derivatives(ll,model,subject,x0,y0,Val(var),args...;kwargs...)
end

@generated function ll_derivatives(ll,model,subject,x0,y0,
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

function mean_(model, _subject, _x0, _vy0, i, args...; kwargs...)
  x_, vals_, dist_ = conditional_nll_ext(model,_subject, _x0, _vy0, args...;kwargs...)
  mean(dist_[1][i])
end

function var_(model, _subject, _x0, _vy0, i, args...; kwargs...)
  x_, vals_, dist_ = conditional_nll_ext(model,_subject, _x0, _vy0, args...;kwargs...)
  var(dist_[1][i])
end

function FIM(m::PKPDModel, subject::Subject, x0, vy0, approx::FOCEI, dist, args...; kwargs...)
  fim = sum(1:length(subject.observations)) do j
    r_inv = inv(var(dist[1][j]))
    res = ll_derivatives(mean_,m,subject, x0, vy0, :η, j, args...;kwargs...)
    f = DiffResults.gradient(res)
    res = ll_derivatives(var_,m,subject, x0, vy0, :η, j, args...;kwargs...)
    del_r = DiffResults.gradient(res)
    f*r_inv*f' + (r_inv*del_r*r_inv*del_r')/2
  end
  fim
end

function FIM(m::PKPDModel, subject::Subject, x0, vy0, approx::FOCE, dist0, args...; kwargs...)
  fim = sum(1:length(subject.observations)) do j
    r_inv = inv(var(dist0[1][j]))
    res = ll_derivatives(mean_,m,subject, x0, vy0, :η, j, args...;kwargs...)
    f = DiffResults.gradient(res)
    f*r_inv*f'
  end
  fim
end
