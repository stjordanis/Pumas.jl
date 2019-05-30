"""
  npde(model, subject, param, randeffs, simulations_count)

To calculate the Normalised Prediction Distribution Errors (NPDE).
"""
function npde(m::PuMaSModel,
              subject::Subject,
              param::NamedTuple,
              randeffs::NamedTuple,
              nsim::Integer)
  y = subject.observations.dv
  sims = [simobs(m, subject, param, randeffs).observed.dv for i in 1:nsim]
  mean_y = mean(sims)
  cov_y = Symmetric(cov(sims))
  Fcov_y = cholesky(cov_y)
  y_decorr = Fcov_y.U'\(y .- mean_y)

  φ = mean(sims) do y_l
    y_decorr_l = Fcov_y\(y_l .- mean_y)
    Int.(y_decorr_l .< y_decorr)
  end

  return quantile.(Normal(), φ)
end

struct SubjectResidual{T1, T2, T3, T4}
  wres::T1
  iwres::T2
  subject::T3
  approx::T4
end
function wresiduals(fpm::FittedPuMaSModel, approx=fpm.approx; nsim=nothing)
  subjects = fpm.data
  if approx == fpm.approx
    vvrandeffs = fpm.vvrandeffs
  else
    # re-estimate under approx
    vvrandeffs = [empirical_bayes(fpm.model, subject, fpm.param, approx) for subject in subjects]
  end
  [wresiduals(fpm, subjects[i], vvrandeffs[i], approx; nsim=nsim) for i = 1:length(subjects)]
end
function wresiduals(fpm, subject::Subject, randeffs, approx; nsim=nothing)
  is_sim = nsim == nothing
  if nsim == nothing
    approx = approx
    wres = wresiduals(fpm.model, subject, fpm.param, randeffs, approx)
    iwres = iwresiduals(fpm.model, subject, fpm.param, randeffs, approx)
  else
    approx = nothing
    wres = nothing
    iwres = eiwres(fpm.model, subject, fpm.param, nsim)
  end

  SubjectResidual(wres, iwres, subject, approx)
end

function DataFrames.DataFrame(vresid::Vector{<:SubjectResidual}; include_covariates=true)
  resid = vresid[1]
  df = DataFrame(id = fill(resid.subject.id, length(resid.subject.time)), wres = resid.wres, iwres = resid.iwres)
  df[:approx] = resid.approx
  if include_covariates
     covariates = resid.subject.covariates
     for (cov, value) in pairs(covariates)
       df[cov] = value
     end
  end
  if length(vresid)>1
    for i = 2:length(vresid)
      resid = vresid[i]
      df_i = DataFrame(id = fill(resid.subject.id, length(resid.subject.time)), wres = resid.wres, iwres = resid.iwres)
      df_i[:approx] = resid.approx
      if include_covariates
         covariates = resid.subject.covariates
         for (cov, value) in pairs(covariates)
           df_i[cov] = value
         end
      end
      df = vcat(df, df_i)
    end
  end
  df
end

function wresiduals(model, subject, param, randeffs, approx::FO)
  wres(model, subject, param, randeffs)
end
function wresiduals(model, subject, param, randeffs, approx::FOCE)
  cwres(model, subject, param, randeffs)
end
function wresiduals(model, subject, param, randeffs, approx::FOCEI)
  cwresi(model, subject, param, randeffs)
end
function iwresiduals(model, subject, param, randeffs, approx::FO)
  iwres(model, subject, param, randeffs)
end
function iwresiduals(model, subject, param, randeffs, approx::FOCE)
  icwres(model, subject, param, randeffs)
end
function iwresiduals(model, subject, param, randeffs, approx::FOCEI)
  icwresi(model, subject, param, randeffs)
end

"""

  wres(model, subject, param[, rfx])

To calculate the Weighted Residuals (WRES).
"""
function wres(m::PuMaSModel,
              subject::Subject,
              param::NamedTuple,
              vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv))
end

"""
  cwres(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals (CWRES).
"""
function cwres(m::PuMaSModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  y = subject.observations.dv
  nl0, dist0 = conditional_nll_ext(m,subject,param, (η=zero(vrandeffs),))
  nl , dist  = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist0.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv) .+ F*vrandeffs)
end

"""
  cwresi(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals with Interaction (CWRESI).
"""

function cwresi(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m, subject, param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv) .+ F*vrandeffs)
end

"""
  pred(model, subject, param[, rfx])

To calculate the Population Predictions (PRED).
"""
function pred(m::PuMaSModel,
              subject::Subject,
              param::NamedTuple,
              vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  nl, dist = conditional_nll_ext(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end


"""
  cpred(model, subject, param[, rfx])

To calculate the Conditional Population Predictions (CPRED).
"""
function cpred(m::PuMaSModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  nl, dist = conditional_nll_ext(m, subject,param, (η=vrandeffs,))
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  return mean.(dist.dv) .- F*vrandeffs
end

"""
  cpredi(model, subject, param[, rfx])

To calculate the Conditional Population Predictions with Interaction (CPREDI).
"""
function cpredi(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  return mean.(dist.dv) .- F*vrandeffs
end

"""
  epred(model, subject, param[, rfx], simulations_count)

To calculate the Expected Simulation based Population Predictions.
"""
function epred(m::PuMaSModel,
               subject::Subject,
               param::NamedTuple,
               randeffs::NamedTuple,
               nsim::Integer)
  sims = [simobs(m, subject, param, randeffs).observed.dv for i in 1:nsim]
  return mean(sims)
end

"""
  iwres(model, subject, param[, rfx])

To calculate the Individual Weighted Residuals (IWRES).
"""
function iwres(m::PuMaSModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist.dv)
end

"""
  icwres(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals (ICWRES).
"""
function icwres(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  y = subject.observations.dv
  nl0, dist0 = conditional_nll_ext(m,subject,param, (η=zero(vrandeffs),))
  nl , dist  = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist0.dv)
end

"""
  icwresi(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals with Interaction (ICWRESI).
"""
function icwresi(m::PuMaSModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  y = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist.dv)
end

"""
  eiwres(model, subject, param[, rfx], simulations_count)

To calculate the Expected Simulation based Individual Weighted Residuals (EIWRES).
"""
function eiwres(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                nsim::Integer)
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m, subject, param, sample_randeffs(m, param))
  sims_sum = (yi .- mean.(dist.dv))./std.(dist.dv)
  for i in 2:nsim
    l, dist = conditional_nll_ext(m, subject, param, sample_randeffs(m, param))
    sims_sum .+= (yi .- mean.(dist.dv))./std.(dist.dv)
  end
  return sims_sum ./ nsim
end

function ipred(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  nl, dist = conditional_nll_ext(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function cipred(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  nl, dist = conditional_nll_ext(m, subject,param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function cipredi(m::PuMaSModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function ηshrinkage(m::PuMaSModel,
                    data::Population,
                    param::NamedTuple,
                    approx::LikelihoodApproximation)
  sd_randeffs = std([empirical_bayes(m, subject, param, approx) for subject in data])
  Ω = Matrix(param.Ω)
  return  1 .- sd_randeffs ./ sqrt.(diag(Ω))
end

function ϵshrinkage(m::PuMaSModel,
                    data::Population,
                    param::NamedTuple,
                    approx::FOCEI,
                    randeffs=[empirical_bayes(m, subject, param, FOCEI()) for subject in data])
  1 - std(vec(VectorOfArray([icwresi(m, subject, param, vrandeffs) for (subject, vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function ϵshrinkage(m::PuMaSModel,
                    data::Population,
                    param::NamedTuple,
                    approx::FOCE,
                    randeffs=[empirical_bayes(m, subject, param, FOCE()) for subject in data])
  1 - std(vec(VectorOfArray([icwres(m, subject, param, vrandeffs) for (subject,vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function StatsBase.aic(m::PuMaSModel,
                       data::Population,
                       param::NamedTuple,
                       approx::LikelihoodApproximation)
  numparam = TransformVariables.dimension(totransform(m.param))
  2*(marginal_nll(m, data, param, approx) + numparam)
end

function StatsBase.bic(m::PuMaSModel,
                       data::Population,
                       param::NamedTuple,
                       approx::LikelihoodApproximation)
  numparam = TransformVariables.dimension(totransform(m.param))
  2*marginal_nll(m, data, param, approx) + numparam*log(sum(t -> length(t.time), data))
end

### Predictions
struct SubjectPrediction{T1, T2, T3, T4}
  pred::T1
  ipred::T2
  subject::T3
  approx::T4
end

function StatsBase.predict(model::PuMaSModel, subject::Subject, param, approx, vrandeffs=empirical_bayes(model, subject, param, approx))
  pred = _predict(model, subject, param, approx, vrandeffs)
  ipred = _ipredict(model, subject, param, approx, vrandeffs)
  SubjectPrediction(pred, ipred, subject, approx)
end

function StatsBase.predict(fpm::FittedPuMaSModel, approx=fpm.approx; nsim=nothing, timegrid=false, newdata=false, useEBEs=true)
  if !useEBEs
    error("Sampling from the omega distribution is not yet implemented.")
  end
  if !(newdata==false)
    error("Using data different than that used to fit the model is not yet implemented.")
  end
  if !(timegrid==false)
    error("Using custom time grids is not yet implemented.")
  end

  subjects = fpm.data

  if approx == fpm.approx
    vvrandeffs = fpm.vvrandeffs
  else
    # re-estimate under approx
    vvrandeffs = [empirical_bayes(fpm.model, subject, fpm.param, approx) for subject in subjects]
  end
  [predict(fpm.model, subjects[i], fpm.param, approx, vvrandeffs[i]) for i = 1:length(subjects)]
end

function _predict(model, subject, param, approx::FO, vrandeffs)
  pred(model, subject, param, vrandeffs)
end
function _ipredict(model, subject, param, approx::FO, vrandeffs)
  ipred(model, subject, param, vrandeffs)
end

function _predict(model, subject, param, approx::Union{FOCE, Laplace}, vrandeffs)
  cpred(model, subject, param, vrandeffs)
end
function _ipredict(model, subject, param, approx::Union{FOCE, Laplace}, vrandeffs)
  cipred(model, subject, param, vrandeffs)
end

function _predict(model, subject, param, approx::Union{FOCEI, LaplaceI}, vrandeffs)
  cpredi(model, subject, param, vrandeffs)
end
function _ipredict(model, subject, param, approx::Union{FOCEI, LaplaceI}, vrandeffs)
  cipredi(model, subject, param, vrandeffs)
end

function epredict(fpm, subject, vrandeffs, nsim::Integer)
  epred(fpm.model, subjects, fpm.param, (η=vrandeffs,), nsim)
end

function DataFrames.DataFrame(vpred::Vector{<:SubjectPrediction}; include_covariates=true)
  # TODO add covariates
  pred = vpred[1]
  df = DataFrame(id = fill(pred.subject.id, length(pred.subject.time)), pred = pred.pred, ipred = pred.ipred)
  df[:approx] = pred.approx
  if include_covariates
     covariates = pred.subject.covariates
     for (cov, value) in pairs(covariates)
       df[cov] = value
     end
  end
  if length(vpred)>1
    for i = 2:length(vpred)
      pred = vpred[i]
      df_i = DataFrame(id = fill(pred.subject.id, length(pred.subject.time)), pred = pred.pred, ipred = pred.ipred)
      df_i[:approx] = pred.approx

      if include_covariates
         covariates = pred.subject.covariates
         for (cov, value) in pairs(covariates)
           df_i[cov] = value
         end
      end
      df = vcat(df, df_i)
    end
  end
  df
end

struct VPC
  Fiftieth
  Fifth_Ninetyfifth
  Simulation_Percentiles
  Observation_Percentiles
  Simulations
  idv
end

function get_strat(data::Population, stratify_on)
  strat_vals = []
  cov_vals = Float64[]
  for i in 1:length(data)
    push!(cov_vals, getproperty(data[i].covariates,stratify_on))
  end
  if length(unique(cov_vals)) <= 4
    return unique(cov_vals)
  else
    return quantile(cov_vals, [0.25,0.5,0.75,1.0])
  end
end

function get_simulation_quantiles(m::PuMaSModel, data::Population, fixeffs::NamedTuple, reps::Integer, dv_, idv_, quantiles, strat_quant, stratify_on)
  sims = []
  pop_quantiles = []
  for i in 1:reps
    sim = simobs(m, data, fixeffs)
    push!(sims, sim)
    quantiles_sim = []
    for t in 1:length(idv_)
      sims_t = [Float64[] for i in 1:length(strat_quant)]
      for j in 1:length(sim.sims)
        for strt in 1:length(strat_quant)
          if  stratify_on == nothing || (length(strat_quant)<4 && stratify_on != nothing && sim.sims[j].subject.covariates[stratify_on] <= strat_quant[strt])
            push!(sims_t[strt], getproperty(sim[j].observed,dv_)[t])
          elseif stratify_on != nothing && sim.sims[j].subject.covariates[stratify_on] <= strat_quant[strt]
            if strt > 1 && sim.sims[j].subject.covariates[stratify_on] > strat_quant[strt-1]
              push!(sims_t[strt], getproperty(sim[j].observed,dv_)[t])
            elseif strt == 1
              push!(sims_t[strt], getproperty(sim[j].observed,dv_)[t])
            end
          end
        end
      end
      push!(quantiles_sim, [quantile(sims_t[strt],quantiles) for strt in 1:length(strat_quant)])
    end 
    push!(pop_quantiles,quantiles_sim)
  end
  pop_quantiles, sims
end

function get_quant_quantiles(pop_quantiles, reps, idv_, quantiles, strat_quant)
  quantile_quantiles = []
  for strt in 1:length(strat_quant)
    quantile_strat = []
    for t in 1:length(idv_)
      quantile_time = []
      for j in 1:length(pop_quantiles[1][t][1])
        quantile_index = Float64[]
        for i in 1:reps
          push!(quantile_index,pop_quantiles[i][t][strt][j])
        end
        push!(quantile_time, quantile(quantile_index,quantiles))
      end
      push!(quantile_strat,quantile_time)
    end
    push!(quantile_quantiles,quantile_strat)
  end
  quantile_quantiles
end

function get_vpc(quantile_quantiles, data, dv_, idv_, sims ,quantiles, strat_quant)
  vpc_strat = VPC[]
  for strt in 1:length(strat_quant)
    fifty_percentiles = []
    fith_ninetyfifth = []
    for i in 1:3
      push!(fifty_percentiles,[j[i][2] for j in quantile_quantiles[strt]])
      push!(fith_ninetyfifth, [(j[i][1],j[i][3]) for j in quantile_quantiles[strt]])
    end
    quantiles_obs = [[] for j in 1:length(quantiles)]
    for t in 1:length(idv_)
      obs_t = [getproperty(data[i].observations, dv_)[t] for i in 1:length(data)]
      for j in 1:length(quantiles)
        push!(quantiles_obs[j], quantile(obs_t,quantiles[j]))
      end
    end 
    push!(vpc_strat, VPC(fifty_percentiles, fith_ninetyfifth, quantile_quantiles, quantiles_obs, sims, idv_))
  end
  vpc_strat
end

function vpc(m::PuMaSModel, data::Population, fixeffs::NamedTuple, reps::Integer;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  # rand_seed = rand()
  # Random.seed!(rand_seed)
  # println("Seed set as $rand_seed")
  vpcs = []
  strat_quants = []
  if stratify_on != nothing
    for strt in stratify_on
      strat_quant = get_strat(data, strt)
      push!(strat_quants , strat_quant)
    end
  else
    push!(strat_quants, 1)
  end

  for dv_ in dv
    if idv == :time
      idv_ = getproperty(data[1], idv)
    else 
      idv_ = getproperty(data[1].covariates, idv)
    end
    stratified_vpc = [] 
    for strat in 1:length(strat_quants)
      
      if stratify_on != nothing
        pop_quantiles, sims = get_simulation_quantiles(m,data,fixeffs,reps, dv_, idv_, quantiles, strat_quants[strat],stratify_on[strat])
      else
        pop_quantiles, sims = get_simulation_quantiles(m,data,fixeffs,reps, dv_, idv_, quantiles, strat_quants[strat],nothing)
      end

      quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])

      vpc_strat = get_vpc(quantile_quantiles, data, dv_, idv_, sims, quantiles, strat_quants[strat])
      push!(stratified_vpc, vpc_strat)
    end

    push!(vpcs , stratified_vpc)
  end
  vpcs
end

function vpc(fpm::FittedPuMaSModel, reps::Integer, data::Population=fpm.data;kwargs...)
  vpc(fpm.model, fpm.data, fpm.param, reps, kwargs...)
end

@recipe function f(vpc::VPC, data::Population)
  if vpc.idv == :time
    t = getproperty(data[1], vpc.idv)
  else 
    t = getproperty(data[1].covariates, vpc.idv)
  end
  xlabel --> string(vpc.idv)
  legend --> false
  lw --> 3
  ylabel --> "Observations"
  title --> "VPC"
  ribbon := vpc.Fifth_Ninetyfifth
  fillalpha := 0.2
  t,[vpc.Fiftieth, vpc.Observation_Percentiles[1:3]]
end

