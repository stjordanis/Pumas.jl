using  AdvancedHMC, MCMCChains

struct SAEMLogDensity{M,D,B,C,R,A,K}
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
    cfg = ForwardDiff.GradientConfig(logdensity, buffer)
    res = DiffResults.GradientResult(buffer)
    SAEMLogDensity(model, data, n, param, buffer, cfg, res, args, kwargs)
  end
  
  function dimension(b::SAEMLogDensity)
    b.dim_rfx * length(b.data)
  end
  
  function logdensity(b::SAEMLogDensity, v::AbstractVector)
    try
      n = b.dim_rfx
      t_param = totransform(b.model.param)
      rfx = b.model.random(b.param)
      t_rfx = totransform(rfx)
      ℓ_rfx = sum(enumerate(b.data)) do (i,subject)
        # compute the random effect density, likelihood and log-Jacobian
        y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view v[((i-1)*n) .+ (1:n)])
        j_y - Pumas.penalized_conditional_nll(b.model, subject, TransformVariables.transform(t_param, b.param), y, b.args...; b.kwargs...)
      end
      ℓ = ℓ_rfx
      return isnan(ℓ) ? -Inf : ℓ
    catch e
      if e isa ArgumentError
        return -Inf
      else
        rethrow(e)
      end
    end
  end
  
  function logdensitygrad(b::SAEMLogDensity, v::AbstractVector)
    ∇ℓ = zeros(size(v))
    try
      n = b.dim_rfx
      fill!(b.buffer, 0.0)
  
      ℓ = DiffResults.value(b.res)
      t_param = totransform(b.model.param)
      param = TransformVariables.transform(t_param, b.param)
      for (i, subject) in enumerate(b.data)
        # to avoid dimensionality problems with ForwardDiff.jl, we split the computation to
        # compute the gradient for each subject individually, then accumulate this to the
        # gradient vector.
  
        function L_rfx(u)
          rfx = b.model.random(param)
          t_rfx = totransform(rfx)
          y, j_y = TransformVariables.transform_and_logjac(t_rfx, @view u[1:n])
          j_y - Pumas.penalized_conditional_nll(b.model, subject, param, y, b.args...; b.kwargs...)
        end
        copyto!(b.buffer, 1, v, (i-1)*n+1, n)
  
        ForwardDiff.gradient!(b.res, L_rfx, b.buffer, b.cfg, Val{false}())
  
        ℓ += DiffResults.value(b.res)
        copyto!(∇ℓ, (i-1)*n+1, DiffResults.gradient(b.res), 1, n)
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
   
  struct SAEMMCMCResults{L<:SAEMLogDensity,C,T}
    loglik::L
    chain::C
    tuned::T
  end
  
  struct SAEM <: LikelihoodApproximation end
  
  function E_step(model::PumasModel, data::Population, param::NamedTuple, nsamples=50, args...; nadapts = 200, kwargs...)
    trf_ident = toidentitytransform(m.param)
    E_step(model, data, TransformVariables.inverse(t_param, fixeffs), nsamples, args...; nadapts = nadapts, kwargs...)
  end

  function E_step(model::PumasModel, population::Population, param::AbstractVector, vrandeffs, nsamples=50, args...; nadapts = 200, kwargs...)
    bayes = SAEMLogDensity(model, population, param, args...;kwargs...)
    dldθ(θ) = logdensitygrad(bayes, θ)
    l(θ) = logdensity(bayes, θ)
    metric = DiagEuclideanMetric(length(vrandeffs))
    h = Hamiltonian(metric, l, dldθ)
    prop = AdvancedHMC.NUTS(Leapfrog(find_good_eps(h, vrandeffs)))
    adaptor = StanHMCAdaptor(nadapts, Preconditioner(metric), NesterovDualAveraging(0.8, prop.integrator.ϵ))
    samples, stats = sample(h, prop, vrandeffs, nsamples, adaptor, nadapts; progress=true)
    b = SAEMMCMCResults(bayes, samples, stats)
    b.chain
  end
  
  function M_step(m, population, fixeffs, randeffs_vec, i, gamma, Qs)
    if i == 1
      Qs[i] = sum(conditional_nll(m, subject, fixeffs, (η = randeff, )) for (subject,randeff) in zip(population,randeffs_vec[1]))
      Qs[i]
    elseif Qs[i] == -Inf
      Qs[i] = M_step(m, population, fixeffs, randeffs_vec, i-1, gamma, Qs) + (gamma)*(sum(sum(conditional_nll(m, subject, fixeffs, (η = randeff,)) - M_step(m, population, fixeffs, randeffs_vec, i-1, gamma, Qs) for (subject,randeff) in zip(population, randeffs)) for randeffs in randeffs_vec[1:i]))
      Qs[i]
    else
      Qs[i] 
    end
  end
  
  function Distributions.fit(m::PumasModel, population::Population, param::NamedTuple, ::SAEM, n::Integer, vrandeffs::AbstractVector,args...; kwargs...)
    eta_vec = []
    trf_ident = toidentitytransform(m.param)
    trf = totransform(m.param)
    vparam = TransformVariables.inverse(trf, param)
    Qs = Float64[]
    for i in 1:n
      eta = E_step(m, population, vparam, vrandeffs, args...; kwargs...)
      dimrandeff = length(m.random(param).params.η)
      for k in 1:length(eta)
        push!(eta_vec,[eta[k][j:j+dimrandeff-1] for j in 1:dimrandeff:length(eta[k])])
      end
      t_param = totransform(m.param)
      Qs = vcat(Qs,[-Inf for i in 1:length(eta)])
      cost = fixeffs -> M_step(m, population, TransformVariables.transform(trf, fixeffs), eta_vec, length(eta_vec), 0.9, Qs)
      # cost = fixeffs -> M_step(m, population, TransformVariables.transform(trf, fixeffs), eta_vec[end-length(eta)+1:end], length(eta), 0.00001, [-Inf for i in 1:length(eta)])
      vparam = Optim.minimizer(Optim.optimize(cost, vparam, Optim.BFGS()))
      vrandeffs = eta[end]
    end
    TransformVariables.transform(trf, vparam)
  end