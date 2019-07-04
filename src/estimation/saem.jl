using LogDensityProblems, DynamicHMC
import LogDensityProblems: AbstractLogDensityProblem, dimension, logdensity

struct SAEMLogDensity{M,D,B,C,R,A,K} <: LogDensityProblems.AbstractLogDensityProblem
    model::M
    data::D
    dim_rfx::Int
    param
    buffer::B
    cfg::C
    res::R
    args::A
    kwargs::K
  end
  
  function SAEMLogDensity(model, data, fixeffs, args...;kwargs...)
    n = dim_rfx(model)
    buffer = zeros(n)
    param = fixeffs
    cfg = ForwardDiff.GradientConfig(LogDensityProblems.logdensity, buffer)
    res = DiffResults.GradientResult(buffer)
    SAEMLogDensity(model, data, n, param, buffer, cfg, res, args, kwargs)
  end
  
  function LogDensityProblems.dimension(b::SAEMLogDensity)
    b.dim_rfx * length(b.data)
  end
  
  function LogDensityProblems.logdensity(::Type{LogDensityProblems.Value}, b::SAEMLogDensity, v::AbstractVector)
      try
        param = b.param
        t_param = totransform(b.model.param)
        # compute the prior density and log-Jacobian
        x, j_x = TransformVariables.transform_and_logjac(t_param, param)
        ℓ_param = _lpdf(param, x) + j_x
    
        n = b.dim_rfx
        rfx = b.model.random(x)
        t_rfx = totransform(rfx)
        ℓ_rfx = sum(enumerate(b.data)) do (i,subject)
          # compute the random effect density, likelihood and log-Jacobian
          y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view v[(i-1)*n .+ (1:n)])
          j_y - PuMaS.penalized_conditional_nll(b.model, subject, x, y, b.args...; b.kwargs...)
        end
        ℓ = ℓ_param + ℓ_rfx
        return LogDensityProblems.Value(isnan(ℓ) ? -Inf : ℓ/ℓ_param)
      catch e
        if e isa ArgumentError
          return LogDensityProblems.Value(-Inf)
        else
          rethrow(e)
        end
      end
    end
  
  function LogDensityProblems.logdensity(::Type{LogDensityProblems.ValueGradient}, b::SAEMLogDensity, v::AbstractVector)
    ∇ℓ = zeros(size(v))
    try
      param = b.param
      n = b.dim_rfx
      t_param = totransform(b.model.param)
      fill!(b.buffer, 0.0)
  
      ℓ = 0.0
  
      for (i, subject) in enumerate(b.data)
        # to avoid dimensionality problems with ForwardDiff.jl, we split the computation to
        # compute the gradient for each subject individually, then accumulate this to the
        # gradient vector.
  
        function L_rfx(u)
          rfx = b.model.random(TransformVariables.transform(t_param,param))
          t_rfx = totransform(rfx)
          y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view u[(1:n)])
          j_y - PuMaS.penalized_conditional_nll(b.model, subject, TransformVariables.transform(t_param,param), y, b.args...; b.kwargs...)
        end
        copyto!(b.buffer, 1, v, (i-1)*n+1, n)
        ForwardDiff.gradient!(b.res, L_rfx, b.buffer, b.cfg, Val{false}())
  
        ℓ += DiffResults.value(b.res)
        copyto!(∇ℓ, (i-1)*n+1, DiffResults.gradient(b.res), 1, n)
      end
      return LogDensityProblems.ValueGradient(isnan(ℓ) ? -Inf : ℓ, ∇ℓ)
    catch e
      if e isa ArgumentError
        return LogDensityProblems.ValueGradient(-Inf, ∇ℓ)
      else
        rethrow(e)
      end
    end
  end
  
  struct SAEMMCMCResults{L<:SAEMLogDensity,C,T}
    loglik::L
    chain::C
    tuned::T
  end
  
  struct SAEM <: LikelihoodApproximation end
  
function Distributions.fit(model::PuMaSModel, data::Population, fixeffs::NamedTuple, ::SAEM,
                             nsamples=5000, args...; kwargs...)
    t_param = totransform(model.param)
    fit(model, data, TransformVariables.inverse(t_param, fixeffs), SAEM(), nsamples, args...; kwargs...)
  end

  function Distributions.fit(model::PuMaSModel, data::Population, fixeffs::AbstractVector, ::SAEM,
                             nsamples=5000, args...; kwargs...)
    bayes = SAEMLogDensity(model, data, fixeffs, args...;kwargs...)
    chain,tuned = NUTS_init_tune_mcmc(bayes, nsamples)
    SAEMMCMCResults(bayes, chain, tuned)
  end

  function E_step(m::PuMaSModel, population::Population, param::NamedTuple, approx)
    b = fit(m, population, param, SAEM(), 50)
    eta_vec = mean(map(get_position, b.chain))
  end
  
  function M_step(m, population, fixeffs, randeffs_vec, i, gamma, Qs, approx)
    if i == 1
      Qs[i] = sum(-conditional_nll(m, subject, fixeffs, (η = randeff, ), approx) for (subject,randeff) in zip(population,randeffs_vec[1]))
    elseif Qs[i] == -Inf
      Qs[i] = M_step(m, population, fixeffs, randeffs_vec, i-1, gamma, Qs, approx) - (gamma/length(randeffs_vec))*(sum(sum(conditional_nll(m, subject, fixeffs, (η = randeff, ), approx) - M_step(m, population, fixeffs, randeffs_vec, i-1, gamma, Qs, approx) for (subject,randeff) in zip(population, randeffs)) for randeffs in randeffs_vec[1:i]))
    else
      Qs[i] 
    end
  end
  
  function SAEM_calc(m::PuMaSModel, population::Population, param::NamedTuple, n, approx)
    eta_vec = []
    for i in 1:n
      eta = E_step(m, population, param, approx)
      dimrandeff = length(m.random(param).params.η)
      push!(eta_vec,[eta[j:j+dimrandeff-1] for j in 1:dimrandeff:length(eta)])
      t_param = totransform(m.param)
      cost = fixeffs -> M_step(m, population, TransformVariables.transform(t_param, fixeffs), eta_vec, length(eta_vec), 0.1, [-Inf for i in 1:length(eta_vec)], approx)
      param = TransformVariables.transform(t_param, Optim.minimizer(Optim.optimize(cost, TransformVariables.inverse(t_param,param), Optim.BFGS())))
      println(param)
    end
    param
  end