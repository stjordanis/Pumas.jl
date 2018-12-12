import DiffResults: DiffResult

Base.@pure flattentype(t) = NamedTuple{fieldnames(typeof(t)), NTuple{length(t), eltype(eltype(t))}}

"""
    conditional_nll(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
function conditional_nll(m::PKPDModel, subject::Subject, args...; extended_return = false, kwargs...)
  obstimes = [obs.time for obs in subject.observations]
  isempty(obstimes) && throw(ArgumentError("obstimes is not specified."))
  derived = derivedfun(m,subject,args...;kwargs...)
  vals, derived_dist = derived(obstimes) # the second component is distributions
  typ = flattentype(derived_dist)
  n = length(derived_dist)
  idx = 1
  x = sum(subject.observations) do obs
    if eltype(derived_dist) <: Array
      l = _lpdf(typ(ntuple(i->derived_dist[i][idx], n)), obs.val)
    else
      l = _lpdf(derived_dist, obs.val)
    end
    idx += 1
    return l
  end
  if extended_return
    return -x, vals, derived_dist
  else
    return -x
  end
end

"""
    penalized_conditional_nll(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as [`conditional_nll`](@ref), except that it incorporates the penalty from the prior distribution of the random effects.
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


abstract type Approximation end
struct Laplace <: Approximation end

function rfx_estimate(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::Laplace, args...; kwargs...)
  rfxset = m.random(x0)
  p = TransformVariables.dimension(totransform(rfxset))
  Optim.minimizer(Optim.optimize(penalized_conditional_nll_fn(m, subject, x0, args...; kwargs...), zeros(p), BFGS(); autodiff=:forward))
end


function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, vy0::AbstractVector, approx::Laplace, args...;
                      kwargs...)
  diffres = DiffResults.HessianResult(vy0)
  penalized_conditional_nll!(diffres, m, subject, x0, vy0, args...; hessian=true, kwargs...)  
  g, m, W = DiffResults.value(diffres),DiffResults.gradient(diffres),DiffResults.hessian(diffres)
  CW = cholesky!(W)
  p = length(vy0)
  g - (p*log(2π) - logdet(CW) + dot(m,CW\m))/2
end

function marginal_nll(m::PKPDModel, subject::Subject, x0::NamedTuple, approx::Approximation, args...;
                      kwargs...)
  vy0 = rfx_estimate(m, subject, x0, approx, args...; kwargs...)
  marginal_nll(m, subject, x0, vy0, approx, args...; kwargs...)
end
function marginal_nll(m::PKPDModel, data::Population, args...;
                      kwargs...)
  sum(subject -> marginal_nll(m, subject, args...; kwargs...), data.subjects)
end


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
