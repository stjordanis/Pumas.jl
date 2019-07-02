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

function _basic_subject_df(subjects)
  _ids = vcat((fill(subject.id, length(subject.time)) for subject in subjects)...)
  _times = vcat((subject.time for subject in subjects)...)
  DataFrame(id=_ids, time=_times)
  end

function add_covariates!(df, subjects)
  # We're assuming that all subjects have the same fields
  for (covariate, value) in pairs(first(subjects).covariates)
    df[covariate] = zero(eltype(value))
  end

  for subject in subjects
    for (covariate, value) in pairs(subject.covariates)
      df[df[:id].==subject.id, covariate] = value
    end
  end
  df
end

function DataFrames.DataFrame(vresid::Vector{<:SubjectResidual}; include_covariates=true)
  subjects = [resid.subject for resid in vresid]
  df = _basic_subject_df(subjects)
  if include_covariates
    add_covariates!(df, subjects)
  end

  df[:wres] = vcat((resid.wres for resid in vresid)...)
  df[:iwres] = vcat((resid.iwres for resid in vresid)...)
  df[:wres_approx] = vcat((fill(resid.approx, length(resid.subject.time)) for resid in vresid)...)

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
  subjects = [pred.subject for pred in vpred]
  df = _basic_subject_df(subjects)
  if include_covariates
    add_covariates!(df, subjects)
  end

  df[:pred] = vcat((pred.pred for pred in vpred)...)
  df[:ipred] = vcat((pred.ipred for pred in vpred)...)
  df[:pred_approx] = vcat((fill(pred.approx, length(pred.subject.time)) for pred in vpred)...)

  df
end

struct VPC_QUANT
  Fiftieth
  Fifth_Ninetyfifth
  Simulation_Percentiles
  Observation_Percentiles
end

struct VPC_STRAT
  vpc_quant::Vector{VPC_QUANT}
  strat
end

struct VPC_DV
  vpc_strat::Vector{VPC_STRAT}
  dv
end

struct VPC
  vpc_dv::Vector{VPC_DV}
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

function get_simulation_quantiles(sims, reps::Integer, dv_, idv_, quantiles, strat_quant, stratify_on)
  pop_quantiles = []
  for i in 1:reps
    sim = sims[i]
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
  pop_quantiles
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

function get_vpc(quantile_quantiles, data, dv_, idv_, sims ,quantiles, strat_quant, stratify_on)
  vpc_strat = VPC_QUANT[]
  for strt in 1:length(strat_quant)
    fifty_percentiles = []
    fith_ninetyfifth = []
    for i in 1:3
      push!(fifty_percentiles,[j[i][2] for j in quantile_quantiles[strt]])
      push!(fith_ninetyfifth, [(j[i][1],j[i][3]) for j in quantile_quantiles[strt]])
    end
    quantiles_obs = [[] for j in 1:length(quantiles)]
    for t in 1:length(idv_)
      if dv_ in keys(data[1].observations)
        obs_t = [getproperty(data[i].observations, dv_)[t] for i in 1:length(data)]
        for j in 1:length(quantiles)
          push!(quantiles_obs[j], quantile(obs_t,quantiles[j]))
        end
      else
        quantiles_obs = nothing
      end
    end
    push!(vpc_strat, VPC_QUANT(fifty_percentiles, fith_ninetyfifth, quantile_quantiles, quantiles_obs))
  end
  VPC_STRAT(vpc_strat, stratify_on)
end

function vpc(m::PuMaSModel, data::Population, fixeffs::NamedTuple, reps::Integer;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  # rand_seed = rand()
  # Random.seed!(rand_seed)
  # println("Seed set as $rand_seed")
  vpcs = VPC_DV[]
  strat_quants = []
  if stratify_on != nothing
    for strt in stratify_on
      strat_quant = get_strat(data, strt)
      push!(strat_quants , strat_quant)
    end
  else
    push!(strat_quants, 1)
  end

  if idv == :time
    idv_ = getproperty(data[1], idv)
  else
    idv_ = getproperty(data[1].covariates, idv)
  end

  sims = []
  for i in 1:reps
    sim = simobs(m, data, fixeffs)
    push!(sims, sim)
  end

  for dv_ in dv
    stratified_vpc = VPC_STRAT[]
    for strat in 1:length(strat_quants)

      if stratify_on != nothing
        pop_quantiles = get_simulation_quantiles(sims, reps, dv_, idv_, quantiles, strat_quants[strat],stratify_on[strat])
        quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])
        vpc_strat = get_vpc(quantile_quantiles, data, dv_, idv_, sims, quantiles, strat_quants[strat], stratify_on[strat])
      else
        pop_quantiles = get_simulation_quantiles(sims, reps, dv_, idv_, quantiles, strat_quants[strat],nothing)
        quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])
        vpc_strat = get_vpc(quantile_quantiles, data, dv_, idv_, sims, quantiles, strat_quants[strat], nothing)
      end


      push!(stratified_vpc, vpc_strat)
    end
    push!(vpcs , VPC_DV(stratified_vpc, dv_))
  end
  VPC(vpcs, sims, idv)
end

function vpc(fpm::FittedPuMaSModel, reps::Integer, data::Population=fpm.data;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  vpc(fpm.model, fpm.data, fpm.param, reps, quantiles=quantiles, idv=idv, dv=dv, stratify_on=stratify_on)
end

function vpc(sims, data::Population;quantiles = [0.05,0.5,0.95], idv = :time, dv = [:dv], stratify_on = nothing)
  # rand_seed = rand()
  # Random.seed!(rand_seed)
  # println("Seed set as $rand_seed")
  if typeof(sims) == VPC
    sims = sims.Simulations
  end

  if idv == :time
    idv_ = getproperty(data[1], idv)
  else
    idv_ = getproperty(data[1].covariates, idv)
  end

  vpcs = []
  strat_quants = []
  if stratify_on != nothing
    for strt in stratify_on
      strat_quant = get_strat(data, strt)
      push!(strat_quants, strat_quant)
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
        pop_quantiles = get_simulation_quantiles(sims, length(sims), dv_, idv_, quantiles, strat_quants[strat], stratify_on[strat])
      else
        pop_quantiles = get_simulation_quantiles(sims, length(sims), dv_, idv_, quantiles, strat_quants[strat], nothing)
      end

      quantile_quantiles = get_quant_quantiles(pop_quantiles,reps,idv_,quantiles, strat_quants[strat])
      vpc_strat = get_vpc(quantile_quantiles, data, dv_, idv_, sims, quantiles, strat_quants[strat], stratify_on)
      push!(stratified_vpc, vpc_strat)
    end
    push!(vpcs , VPC_DV(stratified_vpc, dv_))
  end
  VPC(vpcs, sims, idv)
end

struct SubjectEBES{T1, T2, T3}
  ebes::T1
  subject::T2
  approx::T3
end
function empirical_bayes(fpm::FittedPuMaSModel, approx=fpm.approx)
  subjects = fpm.data

  if approx == fpm.approx
    ebes = fpm.vvrandeffs
    return [SubjectEBES(e, s, approx) for (e, s) in zip(ebes, subjects)]
  else
    # re-estimate under approx
    return [SubjectEBES(empirical_bayes(fpm.model, subject, fpm.param, approx), subject, approx) for subject in subjects]
  end
end

function DataFrames.DataFrame(vebes::Vector{<:SubjectEBES}; include_covariates=true)
  subjects = [ebes.subject for ebes in vebes]
  df = _basic_subject_df(subjects)
  if include_covariates
    add_covariates!(df, subjects)
  end
  for i = 1:length(first(vebes).ebes)
    df[Symbol("ebe_$i")] = vcat((fill(ebes.ebes[i], length(ebes.subject.time)) for ebes in vebes)...)
  end
  df[:ebes_approx] = vcat((fill(ebes.approx, length(ebes.subject.time)) for ebes in vebes)...)

  df
end

struct FittedPuMaSModelInspection{T1, T2, T3, T4}
  o::T1
  pred::T2
  wres::T3
  ebes::T4
end
StatsBase.predict(i::FittedPuMaSModelInspection) = i.pred
wresiduals(i::FittedPuMaSModelInspection) = i.wres
empirical_bayes(i::FittedPuMaSModelInspection) = i.ebes

function inspect(fpm; pred_approx=fpm.approx, infer_approx=fpm.approx,
                    wres_approx=fpm.approx, ebes_approx=fpm.approx)
  print("Calculating: ")
  print("predictions")
  pred = predict(fpm, pred_approx)
  print(", weighted residuals")
  res = wresiduals(fpm, wres_approx)
  print(", empirical bayes")
  ebes = empirical_bayes(fpm, ebes_approx)
  println(". Done.")
  FittedPuMaSModelInspection(fpm, pred, res, ebes)
end
function DataFrames.DataFrame(i::FittedPuMaSModelInspection; include_covariates=true)
  pred_df = DataFrame(i.pred; include_covariates=include_covariates)
  res_df = deletecols!(DataFrame(i.wres; include_covariates=false), :time)
  ebes_df = deletecols!(DataFrame(i.ebes; include_covariates=false), :time)

  df = join(join(pred_df, res_df; on=:id), ebes_df; on=:id)
end

function Base.show(io::IO, mime::MIME"text/plain", pmi::FittedPuMaSModelInspection)
  println(io, "FittedPuMaSModelInspection\n")
  println(io, "Fitting was successful: $(Optim.converged(pmi.o.optim))")

  println(io, "Likehood approximations used for")
  println(io, " * Predictions:        $(first(predict(pmi)).approx)")
  println(io, " * Weighted residuals: $(first(wresiduals(pmi)).approx)")
  println(io, " * Empirical bayes:    $(first(empirical_bayes(pmi)).approx)\n")
end



@recipe function f(vpc::VPC, data::Population)
  if vpc.idv == :time
    t = getproperty(data[1], vpc.idv)
  else
    t = getproperty(data[1].covariates, vpc.idv)
  end
  for i in 1:length(vpc.vpc_dv)
    @series begin
      t,vpc.vpc_dv[i],data, vpc.idv
    end
  end

end

@recipe function f(t, vpc_dv::VPC_DV, data::Population, idv=:time)
  for strt in 1:length(vpc_dv.vpc_strat)
    @series begin
      t, vpc_dv.vpc_strat[strt], data, idv
    end
  end
end

@recipe function f(t, vpc_strt::VPC_STRAT, data::Population, idv=:time)
  layout --> length(vpc_strt.vpc_quant)
  if vpc_strt.strat != nothing
    title --> "Stratified on:"*string(vpc_strt.strat)
  end
  for quant in 1:length(vpc_strt.vpc_quant)
    @series begin
      subplot := quant
      t, vpc_strt.vpc_quant[quant], data, idv
    end
  end
end

@recipe function f(t, vpc_quant::VPC_QUANT, data::Population, idv=:time)
  legend --> false
  lw --> 3
  ribbon := vpc_quant.Fifth_Ninetyfifth
  fillalpha := 0.2
  xlabel --> string(idv)
  ylabel --> "Observations"
  if vpc_quant.Observation_Percentiles != nothing
    t,[vpc_quant.Fiftieth, vpc_quant.Observation_Percentiles[1:3]]
  else
    t, vpc_quant.Fiftieth
  end
end
