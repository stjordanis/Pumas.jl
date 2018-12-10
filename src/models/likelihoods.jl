Base.@pure flattentype(t) = NamedTuple{fieldnames(typeof(t)), NTuple{length(t), eltype(eltype(t))}}

"""
    conditional_ll(m::PKPDModel, subject::Subject, param, rfx, args...; kwargs...)

Compute the full log-likelihood of model `m` for `subject` with parameters `param` and
random effects `rfx`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
function conditional_ll(m::PKPDModel, subject::Subject, args...; extended_return = false, kwargs...)
  obstimes = [obs.time for obs in subject.observations]
  isempty(obstimes) && throw(ArgumentError("obstimes is not specified."))
  derived = derivedfun(m,subject,args...;kwargs...)
  vals, derived_dist = derived(obstimes) # the second component is distributions
  typ = flattentype(derived_dist)
  n = length(derived_dist)
  idx = 1
  x = sum(subject.observations) do obs
    if eltype(derived_dist) <: Array
      l = _lpdf(typ(ntuple(i->derived_dist[i][idx], n)), obs)
    else
      l = _lpdf(derived_dist, obs)
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

function penalized_conditional_nll(m::PKPDModel, subject::Subject, x0, y0, args...;kwargs...)
  Ω = m.random(x0).params.η
  val = conditional_ll(m,subject, x0, y0, args...;kwargs...)
  -val - logpdf(Ω, y0.η)
end

struct Laplace end

function marginal_nll(m::PKPDModel, subject::Subject, x0, y0, approx::Laplace=Laplace(), args...; kwargs...)
    Ω = m.random(x0).params.η
    res = ll_derivatives(penalized_conditional_nll,m,subject, x0, y0, :η, args...;kwargs...)
    g, m, W = DiffResults.value(res),DiffResults.gradient(res),DiffResults.hessian(res)
    p = LinearAlgebra.checksquare(W) # Returns the dimensions
    g - (p*log(2π) - logdet(W) + dot(m,W\m))/2
end

marginal_nll_nonmem(m, subject, x0, y0, args...; kwargs...) =
    2marginal_nll(m, subject, x0, y0, args...;kwargs...) - length(subject.observations)*log(2π)

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
