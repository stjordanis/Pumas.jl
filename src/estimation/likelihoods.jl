import DiffResults: DiffResult

const DEFAULT_RELTOL=1e-6
const DEFAULT_ABSTOL=1e-12

abstract type LikelihoodApproximation end
struct NaivePooled <: LikelihoodApproximation end
struct TwoStage <: LikelihoodApproximation end
struct FO <: LikelihoodApproximation end
struct FOI <: LikelihoodApproximation end
struct FOCE <: LikelihoodApproximation end
struct FOCEI <: LikelihoodApproximation end
struct Laplace <: LikelihoodApproximation end
struct LaplaceI <: LikelihoodApproximation end
struct HCubeQuad <: LikelihoodApproximation end

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

Base.@pure function _intersect_names(an::Tuple{Vararg{Symbol}}, bn::Tuple{Vararg{Symbol}})
    names = Symbol[]
    for n in an
        if Base.sym_in(n, bn)
            push!(names, n)
        end
    end
    (names...,)
end

@generated function _lpdf(ds::NamedTuple{Nds}, xs::NamedTuple{Nxs}) where {Nds, Nxs}
  _names = _intersect_names(Nds, Nxs)
  quote
    names = $_names
    l = _lpdf(getindex(ds, names[1]), getindex(xs, names[1]))
    for i in 2:length(names)
      name = names[i]
      l += _lpdf(getindex(ds, name), getindex(xs, name))
    end
    return l
  end
end

function derived_dist(model::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractArray,
                      args...;
                      # This is the only entry point to the ODE solver for the estimation code
                      # so we need to make sure to set default tolerances here if they haven't
                      # been set elsewhere.
                      reltol=DEFAULT_RELTOL,
                      abstol=DEFAULT_ABSTOL,
                      kwargs...)
  rtrf = totransform(model.random(param))
  randeffs = TransformVariables.transform(rtrf, vrandeffs)
  dist = derived_dist(model, subject, param, randeffs, args...; kwargs...)
end
# inlined beacuse it was taken from conditional_nll_ext below that is inlined
@inline function derived_dist(m::PumasModel,
                              subject::Subject,
                              param::NamedTuple,
                              randeffs::NamedTuple,
                              args...;
                              # This is the only entry point to the ODE solver for the estimation code
                              # so we need to make sure to set default tolerances here if they haven't
                              # been set elsewhere.
                              reltol=DEFAULT_RELTOL,
                              abstol=DEFAULT_ABSTOL,
                              kwargs...)
  # Extract a vector of the time stamps for the observations
  obstimes = subject.time
  isnothing(obstimes) && throw(ArgumentError("no observations for subject"))

  # collate that arguments
  collated = m.pre(param, randeffs, subject)

  # create solution object. By passing saveat=obstimes, we compute the solution only
  # at obstimes such that we can simply pass solution.u to m.derived
  solution = _solve(m, subject, collated, args...; saveat=obstimes, reltol=reltol, abstol=abstol, kwargs...)
  if solution === nothing
    dist = m.derived(collated, solution, obstimes, subject)
  else
    # if solution contains NaN return Inf
    if (solution.retcode != :Success && solution.retcode != :Terminated) ||
      # FIXME! Make this uniform across the two solution types
      any(x->any(isnan,x), solution isa PKPDAnalyticalSolution ? solution(obstimes[end]) : solution.u[end])
      # FIXME! Do we need to make this type stable?
      return nothing
    end

    # extract distributions
    dist = m.derived(collated, solution, obstimes, subject)
  end
  dist
end

"""
    conditional_nll(m::PumasModel, subject::Subject, param, randeffs, args...; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject` with parameters `param` and
random effects `randeffs`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple,
                                 args...;
                                 kwargs...)
    dist = derived_dist(m, subject, param, randeffs, args...; kwargs...)
    conditional_nll(m, subject, param, randeffs, dist)
end

@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple,
                                 dist::Union{NamedTuple, Nothing})

  collated_numtype = numtype(m.pre(param, randeffs, subject))

  if dist isa Nothing
    return collated_numtype(Inf)
  end

  ll = _lpdf(dist, subject.observations)::collated_numtype
  return -ll
end

conditional_nll(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                randeffs::NamedTuple,
                approx::Union{FOI,FOCEI,LaplaceI},
                args...; kwargs...) = conditional_nll(m, subject, param, randeffs, args...; kwargs...)


function conditional_nll(m::PumasModel,
                         subject::Subject,
                         param::NamedTuple,
                         randeffs::NamedTuple,
                         approx::Union{FO,FOCE,Laplace},
                         args...; kwargs...)
  dist = derived_dist(m, subject, param, randeffs, args...;kwargs...)
  l = conditional_nll(m, subject, param, randeffs, dist)

  # If (negative) likehood is infinity or the model is Homoscedastic, we just return l
  if isinf(l) || _is_homoscedastic(dist.dv)
    return l
  else # compute the adjusted (negative) likelihood where the variance is evaluated at η=0
    dist0 = derived_dist(m, subject, param, map(zero, randeffs), args...; kwargs...)
    return _conditional_nll(dist.dv, dist0.dv, subject)
  end
end

# FIXME! Having both a method for randeffs::NamedTuple and vrandeffs::AbstractVector shouldn't really be necessary. Clean it up!
function conditional_nll(m::PumasModel,
                         subject::Subject,
                         param::NamedTuple,
                         vrandeffs::AbstractVector,
                         approx::Union{FO,FOCE,Laplace},
                         args...; kwargs...)
  rfxset = m.random(param)
  randeffs = TransformVariables.transform(totransform(rfxset), vrandeffs)
  conditional_nll(m, subject, param, randeffs, approx, args...; kwargs...)
end

function _conditional_nll(dv::AbstractVector{<:Normal}, dv0::AbstractVector{<:Normal}, subject::Subject)
  x = sum(keys(subject.observations)) do k
    # FIXME! Don't hardcode for dv
    sum(zip(dv, dv0, subject.observations.dv)) do (d, d0, obs)
      _lpdf(Normal(d.μ, d0.σ), obs)
    end
  end
  return -x
end

"""
    penalized_conditional_nll(m::PumasModel, subject::Subject, param::NamedTuple, randeffs::NamedTuple, args...; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as
[`conditional_nll`](@ref), except that it incorporates the penalty from the prior
distribution of the random effects.

Here `param` can be either a `NamedTuple` or a vector (representing a transformation of the
random effects to Cartesian space).
"""
function penalized_conditional_nll(m::PumasModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   randeffs::NamedTuple,
                                   args...;kwargs...)

  # First evaluate the penalty
  rfxset = m.random(param)
  nl_randeffs = -_lpdf(rfxset.params, randeffs)

  # If penalty is too large (likelihood would be Inf) then return without evaluating conditional likelihood
  if nl_randeffs > log(floatmax(Float64))
    return nl_randeffs
  else
    return conditional_nll(m, subject, param, randeffs, args...;kwargs...) + nl_randeffs
  end
end

function penalized_conditional_nll(m::PumasModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   vrandeffs::AbstractVector,
                                   args...;kwargs...)

  rfxset = m.random(param)
  randeffs = TransformVariables.transform(totransform(rfxset), vrandeffs)
  return penalized_conditional_nll(m, subject, param, randeffs, args...; kwargs...)
end

function _initial_randeffs(m::PumasModel, param::NamedTuple)
  rfxset = m.random(param)
  p = TransformVariables.dimension(totransform(rfxset))

  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(param), numtype(param))
  zeros(T, p)
end

"""
empirical_bayes(model, subject, param, approx, ...)
The point-estimate the random effects (being the mode of the empirical Bayes estimate) of a
particular subject at a particular parameter values. The result is returned as a vector
(transformed into Cartesian space).
"""
function empirical_bayes(m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  approx::LikelihoodApproximation,
  args...; kwargs...)
  initial_randeff = _initial_randeffs(m, param)

  return empirical_bayes!(initial_randeff, m, subject, param, approx, args...; kwargs...)
end

function empirical_bayes!(vrandeffs::AbstractVector,
                          m::PumasModel,
                          subject::Subject,
                          param::NamedTuple,
                          ::Union{FO,FOI,HCubeQuad},
                          args...; kwargs...)

  fill!(vrandeffs, 0)
  return vrandeffs
end

function empirical_bayes!(vrandeffs::AbstractVector,
                          m::PumasModel,
                          subject::Subject,
                          param::NamedTuple,
                          approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                          args...;
                          # We explicitly use reltol to compute the right step size for finite difference based gradient
                          reltol=DEFAULT_RELTOL,
                          fdtype=Val{:central}(),
                          fdrelstep=_fdrelstep(m, param, reltol, fdtype),
                          kwargs...)

  cost = _η -> penalized_conditional_nll(m, subject, param, (η=_η,), approx, args...; reltol=reltol, kwargs...)

  vrandeffs .= Optim.minimizer(
    Optim.optimize(
      cost,
      vrandeffs,
      BFGS(
        # Restrict the step sizes allowed by the line search. Large step sizes can make
        # the estimation fail.
        linesearch=Optim.LineSearches.BackTracking(maxstep=1.0),
        ),
      Optim.Options(
        show_trace=false,
        extended_trace=true,
        g_tol=1e-5
      );
      autodiff=:forward))
  return vrandeffs
end

"""
    empirical_bayes_dist(model, subject, param, vrandeffs::AbstractVector, approx, ...)
Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a `MvNormal`.
"""
empirical_bayes_dist

function empirical_bayes_dist(m::PumasModel,
                              subject::Subject,
                              param::NamedTuple,
                              vrandeffs::AbstractVector,
                              approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                              args...; kwargs...)

  Ω = cov(m.random(param).params.η)
  _, _, H = ∂²l∂η²(m, subject, param, vrandeffs, approx, args...; kwargs...)

  # FIXME! For now, we need to convert to Matrix since MvNormal doesn't support SMatrix
  return MvNormal(vrandeffs, inv(cholesky(Symmetric(convert(Matrix, H + inv(Ω))))))
end

"""
    marginal_nll(model, subject, param[, param], approx, ...)
    marginal_nll(model, population, param, approx, ...)

Compute the marginal negative loglikelihood of a subject or dataset, using the integral
approximation `approx`. If no random effect (`param`) is provided, then this is estimated
from the data.

See also [`deviance`](@ref).
"""
function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      randeffs::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...; kwargs...)
  if approx isa NaivePooled
    vrandeffs = []
  else
    rfxset = m.random(param)
    vrandeffs = TransformVariables.inverse(totransform(rfxset), randeffs)
  end
  marginal_nll(m, subject, param, vrandeffs, approx, args...; kwargs...)
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...;
                      kwargs...)
  vrandeffs = empirical_bayes(m, subject, param, approx, args...; kwargs...)
  marginal_nll(m, subject, param, vrandeffs, approx, args...; kwargs...)
end

function marginal_nll(m::PumasModel,
                      # restrict to Vector to avoid distributed arrays taking
                      # this path
                      population::Vector{<:Subject},
                      args...;
                      parallel_type::ParallelType=Threading,
                      kwargs...)

  nll1 = marginal_nll(m, population[1], args...; kwargs...)
  # Compute first subject separately to determine return type and to return
  # early in case the parameter values cause the likelihood to be Inf. This
  # can e.g. happen if the ODE solver can't solve the ODE for the chosen
  # parameter values.
  if isinf(nll1)
    return nll1
  end

  # The different parallel computations are separated out into functions
  # to make it easier to infer the return types
  if parallel_type === Serial
    return sum(subject -> marginal_nll(m, subject, args...; kwargs...), population)
  elseif parallel_type === Threading
    return _marginal_nll_threads(nll1, m, population, args...; kwargs...)
  elseif parallel_type === Distributed # Distributed
    return _marginal_nll_pmap(nll1, m, population, args...; kwargs...)
  else
    throw(ArgumentError("parallel type $parallel_type not implemented"))
  end
end

function _marginal_nll_threads(nll1::T,
                               m::PumasModel,
                               population::Vector{<:Subject},
                               args...;
                               kwargs...)::T where T

  # Allocate array to store likelihood values for each subject in the threaded
  # for loop
  nlls = fill(T(Inf), length(population) - 1)

  # Run threaded for loop for the remaining subjects
  Threads.@threads for i in 2:length(population)
    nlls[i - 1] = marginal_nll(m, population[i], args...; kwargs...)
  end

  return nll1 + sum(nlls)
end

function _marginal_nll_pmap(nll1::T,
                            m::PumasModel,
                            population::Vector{<:Subject},
                            args...;
                            kwargs...)::T where T

  nlls = convert(Vector{T},
                 pmap(subject -> marginal_nll(m, subject, args...; kwargs...),
                 population[2:length(population)]))

    return nll1 + sum(nlls)
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      ::NaivePooled,
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffs))

  # The negative loglikelihood function. There are no random effects.
  conditional_nll(m, subject, param, NamedTuple(), args...;kwargs...)

end
function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      ::FO,
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffs))

  # For FO, the conditional likelihood must be evaluated at η=0
  @assert iszero(vrandeffs)

  # Compute the gradient of the likelihood and Hessian approxmation in the random effect vector η
  # FIXME! Currently we hardcode for dv. Eventually, this should allow for arbitrary dependent variables
  nl, dldη, W = ∂²l∂η²(m, subject, param, vrandeffs, FO(), args...; kwargs...)

  # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  FΩ = cholesky(Ω, check=false)

  # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
  if issuccess(FΩ) && isfinite(nl)
    invFΩ = inv(FΩ)
    return nl + (logdet(FΩ) - dldη'*((invFΩ + W)\dldη) + logdet(invFΩ + W))/2
  else # Ω is numerically singular
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffs::AbstractVector,
                      approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffs))

  nl, _, W = ∂²l∂η²(m, subject, param, vrandeffs, approx, args...; kwargs...)

  # Extract the covariance matrix of the random effects and try computing the Cholesky factorization
  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  FΩ = cholesky(Ω, check=false)

  # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
  if isfinite(nl) && issuccess(FΩ)
    return nl + (logdet(FΩ) + vrandeffs'*(FΩ\vrandeffs) + logdet(cholesky(Symmetric(inv(FΩ) + W))))/2
  else
    # Ω is numerically singular
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      randeffs::AbstractVector,
                      approx::HCubeQuad,
                      low::AbstractVector=-4*sqrt.(var(m.random(param).params.η)),
                      high::AbstractVector=4*sqrt.(var(m.random(param).params.η)),
                      args...; kwargs...)

  -log(hcubature(randeff -> exp(-conditional_nll(m, subject, param, (η = randeff,), args...; kwargs...))*pdf(m.random(param).params.η, randeff), low, high)[1])
end

# deviance is NONMEM-equivalent marginal negative loglikelihood
"""
    deviance(model, subject, param[, param], approx, ...)
    deviance(model, data, param, approx, ...)

Compute the deviance of a subject or dataset:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
StatsBase.deviance(m::PumasModel,
                   subject::Subject,
                   args...; kwargs...) =
    2marginal_nll(m, subject, args...; kwargs...) - length(first(subject.observations))*log(2π)

StatsBase.deviance(m::PumasModel,
                   data::Population,
                   args...; kwargs...) =
    2marginal_nll(m, data, args...; kwargs...) - sum(subject->length(first(subject.observations)), data)*log(2π)
# NONMEM doesn't allow ragged, so this suffices for testing

# Compute the gradient of marginal_nll without solving inner optimization
# problem. This functions follows the approach of Almquist et al. (2015) by
# computing the gradient
#
# dℓᵐ/dθ = ∂ℓᵐ/∂θ + dη/dθ'*∂ℓᵐ/∂η
#
# where ℓᵐ is the marginal likelihood of the subject and dη/dθ is the Jacobian
# of the optimal value of η with respect to the population parameters θ. By
# exploiting that η is computed as the optimum in θ, the Jacobian can be
# computed as
#
# dη/dθ = -∂²ℓᵖ∂η² \ ∂²ℓᵖ∂η∂θ
#
# ℓᵖ is the penalized conditional likelihood function.
function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                subject::Subject,
                                param::NamedTuple,
                                randeffs::NamedTuple,
                                approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                                trf::TransformVariables.TransformTuple,
                                args...;
                                # We explicitly use reltol to compute the right step size for finite difference based gradient
                                reltol=DEFAULT_RELTOL,
                                fdtype=Val{:central}(),
                                fdrelstep=_fdrelstep(model, param, reltol, fdtype),
                                fdabsstep=fdrelstep^2,
                                kwargs...
                                )

  vparam = TransformVariables.inverse(trf, param)

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  ∂ℓᵐ∂θ = DiffEqDiffTools.finite_difference_gradient(
    _param -> marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _param),
      randeffs,
      approx,
      args...;
      reltol=reltol,
      kwargs...
    ),
    vparam,
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  ∂ℓᵐ∂η = DiffEqDiffTools.finite_difference_gradient(
    _η -> marginal_nll(
      model,
      subject,
      param,
      (η=_η,),
      approx,
      args...;
      reltol=reltol,
      kwargs...
    ),
    randeffs.η,
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  # Compute second order derivatives in high precision with ForwardDiff
  ∂²ℓᵖ∂η² = ForwardDiff.hessian(
    _η -> penalized_conditional_nll(model, subject, param, (η=_η,), approx, args...; reltol=reltol, kwargs...),
    randeffs.η
  )

  ∂²ℓᵖ∂η∂θ = ForwardDiff.jacobian(
    _θ -> ForwardDiff.gradient(
      _η -> penalized_conditional_nll(model,
                                      subject,
                                      TransformVariables.transform(trf, _θ),
                                      (η=_η,),
                                      approx,
                                      args...;
                                      reltol=reltol,
                                      kwargs...),
      randeffs.η
    ),
    vparam
  )

  dηdθ = -∂²ℓᵖ∂η² \ ∂²ℓᵖ∂η∂θ

  g .= ∂ℓᵐ∂θ .+ dηdθ'*∂ℓᵐ∂η

  return g
end

function _fdrelstep(model::PumasModel, param::NamedTuple, reltol::AbstractFloat, ::Val{:forward})
  if model.prob isa ExplicitModel
    return sqrt(eps(numtype(param)))
  else
    return max(reltol, sqrt(eps(numtype(param))))
  end
end
function _fdrelstep(model::PumasModel, param::NamedTuple, reltol::AbstractFloat, ::Val{:central})
  if model.prob isa ExplicitModel
    return cbrt(eps(numtype(param)))
  else
    return max(reltol, cbrt(eps(numtype(param))))
  end
end

# Similar to the version for FOCE, FOCEI, Laplace, and LaplaceI
# but much simpler since the expansion point in η is fixed. Hence,
# the gradient is simply the partial derivative in θ
function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                subject::Subject,
                                param::NamedTuple,
                                randeffs::NamedTuple,
                                approx::Union{NaivePooled, FO,FOI,HCubeQuad},
                                trf::TransformVariables.TransformTuple,
                                args...;
                                # We explicitly use reltol to compute the right step size for finite difference based gradient
                                reltol=DEFAULT_RELTOL,
                                fdtype=Val{:central}(),
                                fdrelstep=_fdrelstep(model, param, reltol, fdtype),
                                fdabsstep=fdrelstep^2,
                                kwargs...
                                )

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  g .= DiffEqDiffTools.finite_difference_gradient(
    _param -> marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _param),
      randeffs,
      approx,
      args...;
      reltol=reltol,
      kwargs...
    ),
    TransformVariables.inverse(trf, param),
    typeof(fdtype);
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  return g
end

function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                population::Population,
                                param::NamedTuple,
                                vvrandeffs::AbstractVector,
                                approx::LikelihoodApproximation,
                                trf::TransformVariables.TransformTuple,
                                args...; kwargs...
                                )

  # Zero the gradient
  fill!(g, 0)

  # FIXME! Avoid this allocation
  _g = similar(g)

  for (subject, vrandeffs) in zip(population, vvrandeffs)
    marginal_nll_gradient!(_g, model, subject, param, (η=vrandeffs,), approx, trf, args...; kwargs...)
    g .+= _g
  end

  return g
end

function _conditional_nll_ext_η_gradient(m::PumasModel,
                                         subject::Subject,
                                         param::NamedTuple,
                                         vrandeffs::AbstractVector,
                                         args...; kwargs...)
  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  _conditional_nll_ext =  _η -> begin
      dist = derived_dist(m, subject, param, (η=_η,), args...; kwargs...)
      cnll = conditional_nll(m, subject, param, (η=_η,), dist)
      return cnll, dist
    end
  # Construct vector of dual numbers for the random effects to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(_conditional_nll_ext, vrandeffs)

  ForwardDiff.seed!(cfg.duals, vrandeffs, cfg.seeds)
  cnll, dist = _conditional_nll_ext(cfg.duals)

  return cnll, dist
end

function ∂²l∂η²(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector,
                approx::Union{FO,FOI,FOCE,FOCEI},
                args...; kwargs...)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable
  # per observation while tracking partial derivatives of the random effects
  nl_d, dist_d = _conditional_nll_ext_η_gradient(m, subject, param, vrandeffs, args...; kwargs...)

  if !isfinite(nl_d)
    return ForwardDiff.value(nl_d)::promote_type(numtype(param), numtype(vrandeffs)), nothing, nothing
  end

  return _∂²l∂η²(nl_d, dist_d.dv, m, subject, param, vrandeffs, approx, args...; kwargs...)
end

function _∂²l∂η²(nl_d::ForwardDiff.Dual,
                 dv_d::AbstractVector{<:Union{Normal,LogNormal}},
                 m::PumasModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffs::AbstractVector,
                 ::FO,
                 args...; kwargs...)

  return (ForwardDiff.value(nl_d)::promote_type(numtype(param), numtype(vrandeffs)), _∂²l∂η²(subject.observations.dv, dv_d, FO())...)
end

function _∂²l∂η²(obsdv::AbstractVector, dv::AbstractVector{<:Normal}, ::FO)
  # The dimension of the random effect vector
  nrfx = length(ForwardDiff.partials(first(dv).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)

  # Loop through the distribution vector and extract derivative information
  for j in eachindex(dv)
    obsdvj = obsdv[j]

    # We ignore missing observations when estimating the model
    if ismissing(obsdvj)
      continue
    end

    dvj = dv[j]
    r = ForwardDiff.value(dvj.σ)^2
    f = SVector(ForwardDiff.partials(dvj.μ).values)
    fdr = f/r

    H    += fdr*f'
    dldη += fdr*(obsdvj - ForwardDiff.value(dvj.μ))
  end

  return dldη, H
end

function _∂²l∂η²(obsdv::AbstractVector, dv::AbstractVector{<:LogNormal}, ::FO)
  # The dimension of the random effect vector
  nrfx = length(ForwardDiff.partials(first(dv).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)

  # Loop through the distribution vector and extract derivative information
  for j in eachindex(dv)
    obsdvj = obsdv[j]

    # We ignore missing observations when estimating the model
    if ismissing(obsdvj)
      continue
    end

    r = ForwardDiff.value(dv[j].σ)^2
    f = SVector(ForwardDiff.partials(dv[j].μ).values)
    fdr = f/r
    H += fdr*f'
    dldη += fdr*(log(obsdvj) - ForwardDiff.value(dv[j].μ))
  end

  return dldη, H
end

# Helper function to detect homoscedasticity. For now, it is assumed the input dv vecotr containing
# a vector of distributions with ForwardDiff element types.
function _is_homoscedastic(dv::AbstractVector{<:Union{Normal,LogNormal}})
  # FIXME! Eventually we should support more dependent variables instead of hard coding for dv
  v1 = ForwardDiff.value(first(dv).σ)
  return all(t -> ForwardDiff.value(t.σ) == v1, dv)
end
_is_homoscedastic(::Any) = throw(ArgumentError("Distribution not supported"))

# FIXME! Don't hardcode for dv
function _∂²l∂η²(nl_d::ForwardDiff.Dual,
                 dv_d::AbstractVector{<:Union{Normal,LogNormal}},
                 m::PumasModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffs::AbstractVector,
                 ::FOCE,
                 args...; kwargs...)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation for η=0
  # If the model is homoscedastic, it is not necessary to recompute the variances at η=0
  if _is_homoscedastic(dv_d)
    nl = ForwardDiff.value(nl_d)

    W = _∂²l∂η²(dv_d, first(dv_d), FOCE())
  else # in the Heteroscedastic case, compute the variances at η=0
    dist_0 = derived_dist(m, subject, param, (η=zero(vrandeffs),), args...; kwargs...)

    # Extract the value of the conditional likelihood
    nl = ForwardDiff.value(_conditional_nll(dv_d, dist_0.dv, subject))

    # Compute the Hessian approxmation in the random effect vector η
    # FIXME! Don't hardcode for dv
    W = _∂²l∂η²(dv_d, dist_0.dv, FOCE())
  end
  return nl, nothing, W
end

# Homoscedastic case
function _∂²l∂η²(dv_d::AbstractVector{<:Union{Normal,LogNormal}}, dv0_d::Union{Normal,LogNormal}, ::FOCE)
  # Loop through the distribution vector and extract derivative information
  σ = ForwardDiff.value(dv0_d.σ)
  H = sum(eachindex(dv_d)) do j
    f = SVector(ForwardDiff.partials(dv_d[j].μ).values)/σ
    return f*f'
  end

  return H
end

# Heteroscedastic case
function _∂²l∂η²(dv_d::AbstractVector{<:Normal}, dv0::AbstractVector{<:Normal}, ::FOCE)
  # Loop through the distribution vector and extract derivative information
  H = sum(eachindex(dv_d)) do j
    σ = dv0[j].σ
    f = SVector(ForwardDiff.partials(dv_d[j].μ).values)/σ
    return f*f'
  end

  return H
end

function _∂²l∂η²(dv::AbstractVector{<:Union{Normal,LogNormal}}, ::FOCEI)
  # Loop through the distribution vector and extract derivative information
  H = sum(eachindex(dv)) do j
    dvj   = dv[j]
    r_inv = inv(ForwardDiff.value(dvj.σ^2))
    f     = SVector(ForwardDiff.partials(dvj.μ).values)
    del_r = SVector(ForwardDiff.partials(dvj.σ.^2).values)
    f*r_inv*f' + (r_inv*del_r*r_inv*del_r')/2
  end

  return H
end

# For FOCE we need all the arguments in case we need to recompute the model at η=0 (heteroscedastic case)
# but for FOCE we just pass dv_d on to the actual computational kernel.
_∂²l∂η²(nl_d::ForwardDiff.Dual,
        dv_d::AbstractVector{<:Union{Normal,LogNormal}},
        m::PumasModel,
        subject::Subject,
        param::NamedTuple,
        vrandeffs::AbstractVector,
        ::FOCEI,
        args...; kwargs...) = ForwardDiff.value(nl_d), nothing, _∂²l∂η²(dv_d, FOCEI())


# FIXME! Don't hardcode for dv
function ∂²l∂η²(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector,
                approx::Union{Laplace,LaplaceI},
                args...; kwargs...)

  # Initialize HessianResult for computing Hessian, gradient and value of negative loglikelihood in one go
  diffres = DiffResults.HessianResult(vrandeffs)

  # Compute the derivates
  ForwardDiff.hessian!(diffres,
    _vrandeffs -> conditional_nll(m, subject, param, (η=_vrandeffs,), approx, args...; kwargs...),
  vrandeffs)

#   # Extract the derivatives
  nl, W = DiffResults.value(diffres), DiffResults.hessian(diffres)

  return nl, nothing, W
end

# Fallbacks for a usful error message when distribution isn't supported
_∂²l∂η²(nl_d::ForwardDiff.Dual,
        dv_d::Any,
        m::PumasModel,
        subject::Subject,
        param::NamedTuple,
        vrandeffs::AbstractVector,
        approx::LikelihoodApproximation,
        args...; kwargs...) = throw(ArgumentError("Distribution is current not supported for the $approx approximation. Please consider a different likelihood approximation."))

# Fitting methods
struct FittedPumasModel{T1<:PumasModel,T2<:Population,T3,T4<:LikelihoodApproximation, T5}
  model::T1
  data::T2
  optim::T3
  approx::T4
  vvrandeffs::T5
end

function DEFAULT_OPTIMIZE_FN(cost, p, callback)
  Optim.optimize(
    cost,
    p,
    BFGS(
      linesearch=Optim.LineSearches.BackTracking(),
      # Make sure that step isn't too large by scaling initial Hessian by the norm of the initial gradient
      initial_invH=t -> Matrix(I/norm(Optim.NLSolversBase.gradient(cost)), length(p), length(p))
    ),
    Optim.Options(
      show_trace=false, # Print progress
      store_trace=true,
      extended_trace=true,
      g_tol=1e-3,
      allow_f_increases=true,
      callback=callback
    )
  )
end

function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           approx::LikelihoodApproximation,
                           args...;
                           # optimize_fn should take the arguments cost, p, and callback where cost is a
                           # NLSolversBase.OnceDifferentiable, p is a Vector, and cl is Function. Hence,
                           # optimize_fn should evaluate cost with the NLSolversBase.value and
                           # NLSolversBase.gradient interface. The cl callback should be called once per
                           # outer iteration when applicable but it is not required that the optimization
                           # procedure calls cl. In that case, the estimation of the EBEs will always begin
                           # in zero. In addition, the returned object should support a opt_minimizer method
                           # that returns the optimized parameters.
                           optimize_fn = DEFAULT_OPTIMIZE_FN,
                           kwargs...)

  # Compute transform object defining the transformations from NamedTuple to Vector while applying any parameter restrictions and apply the transformations
  trf = totransform(m.param)
  vparam = TransformVariables.inverse(trf, param)

  # We'll store the random effects estimate in vvrandeffs which allows us to carry the estimates from last
  # iteration and use them as staring values in the next iteration. We also allocate a buffer to store the
  # random effect estimate during an iteration since it might be modified several times during a line search
  # before the new value has been found. We then define a callback which will store values of vvrandeffs_tmp
  # in vvrandeffs once the iteration is done.
  if approx isa NaivePooled
    vvrandeffs     = [[] for subject in population]
    vvrandeffs_tmp = [copy(vrandeffs) for vrandeffs in vvrandeffs]
    cb(state) = false
  else
    vvrandeffs     = [zero(mean(m.random(param).params.η)) for subject in population]
    vvrandeffs_tmp = [copy(vrandeffs) for vrandeffs in vvrandeffs]
    cb = state -> begin
      for i in eachindex(vvrandeffs)
        copyto!(vvrandeffs[i], vvrandeffs_tmp[i])
      end
      return false
    end
  end
  # Define cost function for the optimization
  cost = Optim.NLSolversBase.OnceDifferentiable(
    Optim.NLSolversBase.only_fg!() do f, g, vx
      # The negative loglikelihood function
      # Update the Empirical Bayes Estimates explicitly after each iteration

      # Convert vector to NamedTuple
      x = TransformVariables.transform(trf, vx)

      # Sum up loglikelihood contributions
      nll = sum(zip(population, vvrandeffs, vvrandeffs_tmp)) do (subject, vrandeffs, vrandeffs_tmp)
        # If not FO then compute EBE based on the estimates from last iteration stored in vvrandeffs
        # and store the retult in vvrandeffs_tmp
        if !(approx isa FO || approx isa NaivePooled)
          copyto!(vrandeffs_tmp, vrandeffs)
          empirical_bayes!(vrandeffs_tmp, m, subject, x, approx, args...; kwargs...)
        end
        marginal_nll(m, subject, x, vrandeffs_tmp, approx, args...; kwargs...)
      end

      # Update score
      if g !== nothing
        marginal_nll_gradient!(g, m, population, x, vvrandeffs_tmp, approx, trf, args...; kwargs...)
      end

      return nll
    end,

    # The initial values
    vparam
  )

  # Run the optimization
  o = optimize_fn(cost, vparam, cb)

  # Update the random effects after optimization
  if !(approx isa FO || approx isa NaivePooled)
    for (vrandeffs, subject) in zip(vvrandeffs, population)
      empirical_bayes!(vrandeffs, m, subject, TransformVariables.transform(trf, opt_minimizer(o)), approx, args...; kwargs...)
    end
  end

  return FittedPumasModel(m, population, o, approx, vvrandeffs)
end

function Distributions.fit(m::PumasModel,
                           subject::Subject,
                           param::NamedTuple,
                           args...;
                           kwargs...)
  return fit(m, [subject,], param, NaivePooled(), args...; kwargs...)
end
function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           ::TwoStage,
                           args...;
                           kwargs...)
  return map(x->fit(m, [x,], param, NaivePooled(), args...; kwargs...), population)
end

# error handling for fit(model, subject, param, args...; kwargs...)
function Distributions.fit(model::PumasModel, subject::Subject,
             param::NamedTuple, approx::LikelihoodApproximation, args...; kwargs...)
  throw(ArgumentError("Calling fit on a single subject is not allowed with a likelihood approximation method specified."))
end

opt_minimizer(o::Optim.OptimizationResults) = Optim.minimizer(o)

function Base.getproperty(f::FittedPumasModel{<:Any,<:Any,<:Optim.MultivariateOptimizationResults}, s::Symbol)
  if s === :param
    trf = totransform(f.model.param)
    TransformVariables.transform(trf, opt_minimizer(f.optim))
  else
    return getfield(f, s)
  end
end

marginal_nll(      f::FittedPumasModel) = marginal_nll(f.model, f.data, f.param, f.approx)
StatsBase.deviance(f::FittedPumasModel) = deviance(    f.model, f.data, f.param, f.approx)

function _observed_information(f::FittedPumasModel,
                                ::Val{Score},
                               args...;
                               # We explicitly use reltol to compute the right step size for finite difference based gradient
                               # The tolerance has to be stricter when computing the covariance than during estimation
                               reltol=abs2(DEFAULT_RELTOL),
                               kwargs...) where Score
  # Transformation the NamedTuple of parameters to a Vector
  # without applying any bounds (identity transform)
  trf = toidentitytransform(f.model.param)
  param = f.param
  vparam = TransformVariables.inverse(trf, param)

  fdrelstep_score = _fdrelstep(f.model, f.param, reltol, Val{:central}())
  fdrelstep_hessian = sqrt(_fdrelstep(f.model, f.param, reltol, Val{:central}()))

  # Initialize arrays
  H = zeros(eltype(vparam), length(vparam), length(vparam))
  if Score
    S = copy(H)
    g = similar(vparam, length(vparam))
  else
    S = g = nothing
  end

  # Loop through subject and compute Hessian and score contributions
  for i in eachindex(f.data)
    subject = f.data[i]

    # Compute Hessian contribution and update Hessian
    H .+= DiffEqDiffTools.finite_difference_jacobian(vparam,
                                                     Val{:central};
                                                     relstep=fdrelstep_hessian,
                                                     absstep=fdrelstep_hessian^2) do _j, _param
      param = TransformVariables.transform(trf, _param)
      vrandeffs = empirical_bayes(f.model, subject, param, f.approx, args...; kwargs...)
      marginal_nll_gradient!(_j, f.model, subject, param, (η=vrandeffs,), f.approx, trf, args...;
        reltol=reltol,
        fdtype=Val{:central}(),
        fdrelstep=fdrelstep_hessian,
        fdabsstep=fdrelstep_hessian^2,
        kwargs...)
      return nothing
    end

    if Score
      # Compute score contribution
      vrandeffs = empirical_bayes(f.model, subject, f. param, f.approx, args...; kwargs...)
      marginal_nll_gradient!(g, f.model, subject, f.param, (η=vrandeffs,), f.approx, trf, args...;
        reltol=reltol,
        fdtype=Val{:central}(),
        fdrelstep=fdrelstep_score,
        fdabsstep=fdrelstep_hessian^2,
        kwargs...)

      # Update outer product of scores
      S .+= g .* g'
    end
  end
  return H, S
end

function _expected_information(m::PumasModel,
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

function StatsBase.informationmatrix(f::FittedPumasModel; expected::Bool=true)
  data      = f.data
  model     = f.model
  param     = f.param
  vrandeffs = f.vvrandeffs
  if expected
    return sum(_expected_information(model, data[i], param, (η=vrandeffs[i],), f.approx) for i in 1:length(data))
  else
    return first(_observed_information(f, Val(false)))
  end
end

"""
    vcov(f::FittedPumasModel) -> Matrix

Compute the covariance matrix of the population parameters
"""
function StatsBase.vcov(f::FittedPumasModel, args...; kwargs...)

  # Compute the observed information based on the Hessian (H) and the product of the outer scores (S)
  H, S = _observed_information(f, Val(true), args...; kwargs...)

  # Use generialized eigenvalue decomposition to compute inv(H)*S*inv(H)
  F = eigen(Symmetric(H), Symmetric(S))
  any(t -> t <= 0, F.values) && @warn("Hessian is not positive definite")
  return F.vectors*Diagonal(inv.(abs2.(F.values)))*F.vectors'
end

"""
    stderror(f::FittedPumasModel) -> NamedTuple

Compute the standard errors of the population parameters and return
the result as a `NamedTuple` matching the `NamedTuple` of population
parameters.
"""
StatsBase.stderror(f::FittedPumasModel) = stderror(infer(f))

function Statistics.mean(vfpm::Vector{<:FittedPumasModel})
  names = keys(first(vfpm).param)
  means = []
  for name in names
    push!(means, mean([fpm.param[name] for fpm in vfpm]))
  end
  NamedTuple{names}(means)
end
function Statistics.std(vfpm::Vector{<:FittedPumasModel})
  names = keys(first(vfpm).param)
  stds = []
  for name in names
    push!(stds, std([fpm.param[name] for fpm in vfpm]))
  end
  NamedTuple{names}(stds)
end
function Statistics.var(vfpm::Vector{<:FittedPumasModel})
  names = keys(first(vfpm).param)
  vars = []
  for name in names
    push!(vars, var([fpm.param[name] for fpm in vfpm]))
  end
  NamedTuple{names}(vars)
end

function _E_and_V(m::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector,
                  ::FO,
                  args...; kwargs...)

  y = subject.observations.dv

  dist = derived_dist(m, subject, param, (η=vrandeffs,))
  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  F = ForwardDiff.jacobian(s -> mean.(derived_dist(m, subject, param, (η=s,)).dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return mean.(dist.dv), V
end

function _E_and_V(m::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector,
                  ::FOCE,
                  args...; kwargs...)

  y = subject.observations.dv
  dist0 = derived_dist(m, subject, param, (η=zero(vrandeffs),))
  dist  = derived_dist(m, subject, param, (η=vrandeffs,))

  # We extract the covariance of random effect like this because Ω might not be a parameters of param if it is fixed.
  Ω = cov(m.random(param).params.η)
  F = ForwardDiff.jacobian(s -> mean.(derived_dist(m, subject, param, (η=s,)).dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist0.dv)))

  return mean.(dist.dv) .- F*vrandeffs, V

end

# Some type piracy for the time being
Distributions.MvNormal(D::Diagonal) = MvNormal(PDiagMat(D.diag))

struct FittedPumasModelInference{T1, T2, T3}
  fpm::T1
  vcov::T2
  level::T3
end

"""
    infer(fpm::FittedPumasModel) -> FittedPumasModelInference

Compute the `vcov` matrix and return a struct used for inference
based on the fitted model `fpm`.
"""
function infer(fpm::FittedPumasModel, args...; level = 0.95, kwargs...)
  print("Calculating: variance-covariance matrix")
  _vcov = vcov(fpm, args...; kwargs...)
  println(". Done.")
  FittedPumasModelInference(fpm, _vcov, level)
end
function StatsBase.stderror(pmi::FittedPumasModelInference)
  ss = sqrt.(diag(pmi.vcov))
  trf = tostderrortransform(pmi.fpm.model.param)
  return TransformVariables.transform(trf, ss)
end

# empirical_bayes_dist for FiitedPumas
function empirical_bayes_dist(fpm::FittedPumasModel)
  StructArray(
    (η=map(zip(fpm.data, fpm.vvrandeffs)) do (subject, vrandeffs)
      empirical_bayes_dist(fpm.model, subject, fpm.param, vrandeffs, fpm.approx)
    end,)
    )
end
