function dim_param(m::PumasModel)
  t_param = totransform(m.param)
  TransformVariables.dimension(t_param)
end

function dim_rfx(m::PumasModel)
  x = init_param(m)
  rfx = m.random(x)
  t_rfx = totransform(rfx)
  TransformVariables.dimension(t_rfx)
end

# This object wraps the model, data and some pre-allocated buffers to match the necessary interface for DynamicHMC.jl
struct BayesLogDensity{M,D,B,C,R,A,K}
  model::M
  data::D
  dim_param::Int
  dim_rfx::Int
  buffer::B
  cfg::C
  res::R
  args::A
  kwargs::K
end

function BayesLogDensity(model, data, args...;kwargs...)
  m = dim_param(model)
  n = dim_rfx(model)
  buffer = zeros(m + n)
  cfg = ForwardDiff.GradientConfig(logdensity, buffer)
  res = DiffResults.GradientResult(buffer)
  BayesLogDensity(model, data, m, n, buffer, cfg, res, args, kwargs)
end

function dimension(b::BayesLogDensity)
  b.dim_param + b.dim_rfx * length(b.data)
end

function logdensity(b::BayesLogDensity, v::AbstractVector)
  try
    m = b.dim_param
    param = b.model.param
    t_param = totransform(param)
    # compute the prior density and log-Jacobian
    x, j_x = TransformVariables.transform_and_logjac(t_param, @view v[1:m])
    ℓ_param = _lpdf(param.params, x) + j_x

    n = b.dim_rfx
    rfx = b.model.random(x)
    t_rfx = totransform(rfx)
    ℓ_rfx = sum(enumerate(b.data)) do (i,subject)
      # compute the random effect density, likelihood and log-Jacobian
      y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view v[(m+(i-1)*n) .+ (1:n)])
      j_y - Pumas.penalized_conditional_nll(b.model, subject, x, y, b.args...; b.kwargs...)
    end
    ℓ = ℓ_param + ℓ_rfx
    return isnan(ℓ) ? -Inf : ℓ
  catch e
    if e isa ArgumentError
      return -Inf
    else
      rethrow(e)
    end
  end
end

function logdensitygrad(b::BayesLogDensity, v::AbstractVector)
  ∇ℓ = zeros(size(v))
  try
    param = b.model.param
    t_param = totransform(param)
    m = b.dim_param
    n = b.dim_rfx

    function L_param(u)
      x, j_x = TransformVariables.transform_and_logjac(t_param, u)
      _lpdf(param.params, x) + j_x
    end
    fill!(b.buffer, 0.0)
    copyto!(b.buffer, 1, v, 1, m)

    ForwardDiff.gradient!(b.res, L_param, b.buffer, b.cfg, Val{false}())

    ℓ = DiffResults.value(b.res)
    ∇ℓ[1:m] .= @view DiffResults.gradient(b.res)[1:m]

    for (i, subject) in enumerate(b.data)
      # to avoid dimensionality problems with ForwardDiff.jl, we split the computation to
      # compute the gradient for each subject individually, then accumulate this to the
      # gradient vector.

      function L_rfx(u)
        x = TransformVariables.transform(t_param, @view u[1:m])
        rfx = b.model.random(x)
        t_rfx = totransform(rfx)
        y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view u[m .+ (1:n)])
        j_y - Pumas.penalized_conditional_nll(b.model, subject, x, y, b.args...; b.kwargs...)
      end
      copyto!(b.buffer, m+1, v, m+(i-1)*n+1, n)

      ForwardDiff.gradient!(b.res, L_rfx, b.buffer, b.cfg, Val{false}())

      ℓ += DiffResults.value(b.res)
      ∇ℓ[1:m] .+= @view DiffResults.gradient(b.res)[1:m]
      copyto!(∇ℓ, m+(i-1)*n+1, DiffResults.gradient(b.res), m+1, n)
    end
    return isnan(ℓ) ? -Inf : ℓ, ∇ℓ
  catch e
    if e isa ArgumentError
      return -Inf, ∇ℓ
    else
      rethrow(e)
    end
  end
end

# results wrapper object
struct BayesMCMCResults{L<:BayesLogDensity,C,T}
  loglik::L
  chain::C
  tuned::T
end

struct BayesMCMC <: LikelihoodApproximation end

# function Distributions.fit(model::PumasModel, data::Population, ::BayesMCMC,
#                            args...; nsamples=5000, kwargs...)
#   bayes = BayesLogDensity(model, data, args...;kwargs...)
#   chain,tuned = NUTS_init_tune_mcmc(bayes, nsamples)
#   BayesMCMCResults(bayes, chain, tuned)
# end

function Distributions.fit(model::PumasModel, data::Population, ::BayesMCMC, θ_init,
  args...; n_adapts=2000,n_samples=10000, kwargs...)
  bayes = BayesLogDensity(model, data, args...;kwargs...)
  dldθ(θ) = logdensitygrad(bayes, θ)
  l(θ) = logdensity(bayes, θ)
  metric = DiagEuclideanMetric(length(θ_init))
  h = Hamiltonian(metric, l, dldθ)
  prop = AdvancedHMC.NUTS(Leapfrog(find_good_eps(h, θ_init)))
  adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, prop.integrator.ϵ))
  samples, stats = sample(h, prop, θ_init, n_samples, adaptor, n_adapts; progress=true)
  BayesMCMCResults(bayes, samples, stats)
end

# remove unnecessary PDMat wrappers
_clean_param(x) = x
_clean_param(x::PDMat) = x.mat
_clean_param(x::NamedTuple) = map(_clean_param, x)

# "unzip" the results via a StructArray
function param_values(b::BayesMCMCResults)
  param = b.loglik.model.param
  t_param = totransform(param)
  StructArray([_clean_param(TransformVariables.transform(t_param, c)) for c in b.chain])
end

function param_mean(b::BayesMCMCResults)
  vals = param_values(b)
  map(mean, StructArrays.fieldarrays(vals))
end
function param_var(b::BayesMCMCResults)
  vals = param_values(b)
  map(var, StructArrays.fieldarrays(vals))
end
function param_std(b::BayesMCMCResults)
  vals = param_values(b)
  map(std, StructArrays.fieldarrays(vals))
end
