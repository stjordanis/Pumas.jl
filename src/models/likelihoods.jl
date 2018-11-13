"""
_likelihood(err, obs)

Computes the log-likelihood between the err and obs, only using err terms that
also have observations, and assuming the Dirac distribution for any err terms
that are numbers.
"""
function _likelihood(err::T, obs) where {T}
  syms =  fieldnames(T) ∩ fieldnames(typeof(obs.val))
  sum(map((d,x) -> isnan(x) ? zval(d) : _lpdf(d,x), (getproperty(err,x) for x in syms), (getproperty(obs.val,x) for x in syms)))
end

Base.@pure flattentype(t) = NamedTuple{fieldnames(typeof(t)), NTuple{length(t), eltype(eltype(t))}}

"""
    likelihood(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the full log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
function conditional_loglikelihood(m::PKPDModel, subject::Subject, args...; extended_return = false, kwargs...)
  obstimes = [obs.time for obs in subject.observations]
  isempty(obstimes) && throw(ArgumentError("obstimes is not specified."))
  derived = derivedfun(m,subject,args...;kwargs...)
  vals, derived_dist = derived(obstimes) # the second component is distributions
  typ = flattentype(derived_dist)
  n = length(derived_dist)
  idx = 1
  x = sum(subject.observations) do obs
    if eltype(derived_dist) <: Array
      l = _likelihood(typ(ntuple(i->derived_dist[i][idx], n)), obs)
    else
      l = _likelihood(derived_dist, obs)
    end
    idx += 1
    return l
  end
  if extended_return
    return x, vals, derived_dist
  else
    return x
  end
end

function peanalized_conditional_loglikelihood(m::PKPDModel, subject::Subject, x0, y0, args...;kwargs...)
  Ω = m.random(x0).dists.η
  val = conditional_loglikelihood(m,subject, x0, y0, args...;kwargs...)
  g = -val - logpdf(Ω, y0.η)
end

struct Laplace end

function marginal_loglikelihood(m::PKPDModel, subject::Subject, x0, y0, approx::Laplace=Laplace(), args...; kwargs...)
    Ω = m.random(x0).dists.η
    g, m, W = conditional_loglikelihood_derivatives(m,subject, x0, y0, :η, args...;kwargs...)
    p = LinearAlgebra.checksquare(W) # Returns the dimensions
    -g - (p*log(2π) - logdet(W) + dot(m,W\m))/2
end

marginal_loglikelihood_nonmem(m, subject, x0, args...; kwargs...) =
    -2 * marginal_loglikelihood(m, subject, x0, args...;kwargs...) + length(subject.observations)*log(2*pi)


"""
In named tuple nt, replace the value x.var by y
"""
@generated function Base.setindex(x::NamedTuple,y,v::Val)
  k = first(v.parameters)
  k ∉ x.names ? :x : :( (x..., $k=y) )
end

function generate_enclosed_likelihood(model,subject,x0,y0,v, args...; kwargs...)
  f = function (z)
    _x0 = setindex(x0,z,v)
    _y0 = setindex(y0,z,v)
    peanalized_conditional_loglikelihood(model, subject, _x0, _y0, args...; kwargs...)
  end
end

function conditional_loglikelihood_derivatives(model,subject,x0,y0,var::Symbol,transform=false,
                                args...; kwargs...)
  conditional_loglikelihood_derivatives(model,subject,x0,y0,Val(var),args...;
                         transform=transform,
                         kwargs...)
end

@generated function conditional_loglikelihood_derivatives(model,subject,x0,y0,
                                           v::Val{var}, args...;
                                           hessian_required = true,
                                           transform=false, kwargs...) where var
  var ∉ fieldnames(x0) ∪ fieldnames(y0) && error("Variable name not recognized")
  valex = var ∈ fieldnames(x0) ? :(x0.$var) : :(y0.$var)
  quote
    f = generate_enclosed_likelihood(model,subject,x0,y0,v,args...;kwargs...)
    if hessian_required
      result = DiffResults.HessianResult($valex)
      result = ForwardDiff.hessian!(result, f, $valex)
      return DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    else
      result = DiffResults.GradientResult($valex)
      result = ForwardDiff.gradient!(result, f, $valex)
      return DiffResults.value(result), DiffResults.gradient(result)
    end
  end
end
