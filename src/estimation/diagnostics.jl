"""
  npde(model, subject, param, randeffs, simulations_count)

To calculate the Normalised Prediction Distribution Errors (NPDE).
"""
function npde(m::PumasModel,
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
function wresiduals(fpm::FittedPumasModel, approx=fpm.approx; nsim=nothing)
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
  if !isa(first(subjects).covariates, Nothing)
    for (covariate, value) in pairs(first(subjects).covariates)
      df[!,covariate] .= value
    end

    for subject in subjects
      for (covariate, value) in pairs(subject.covariates)
        df[df[!,:id].==subject.id, covariate] .= value
      end
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

  df[!,:wres] .= vcat((resid.wres for resid in vresid)...)
  df[!,:iwres] .= vcat((resid.iwres for resid in vresid)...)
  df[!,:wres_approx] .= vcat((fill(resid.approx, length(resid.subject.time)) for resid in vresid)...)

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
function wres(m::PumasModel,
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
function cwres(m::PumasModel,
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

function cwresi(m::PumasModel,
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
function pred(m::PumasModel,
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
function cpred(m::PumasModel,
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
function cpredi(m::PumasModel,
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
function epred(m::PumasModel,
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
function iwres(m::PumasModel,
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
function icwres(m::PumasModel,
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
function icwresi(m::PumasModel,
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
function eiwres(m::PumasModel,
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

function ipred(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FO()))
  nl, dist = conditional_nll_ext(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function cipred(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCE()))
  nl, dist = conditional_nll_ext(m, subject,param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function cipredi(m::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractVector=empirical_bayes(m, subject, param, FOCEI()))
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return mean.(dist.dv)
end

function ηshrinkage(m::PumasModel,
                    data::Population,
                    param::NamedTuple,
                    approx::LikelihoodApproximation)
  sd_randeffs = std([empirical_bayes(m, subject, param, approx) for subject in data])
  Ω = Matrix(param.Ω)
  return  1 .- sd_randeffs ./ sqrt.(diag(Ω))
end

function ϵshrinkage(m::PumasModel,
                    data::Population,
                    param::NamedTuple,
                    approx::FOCEI,
                    randeffs=[empirical_bayes(m, subject, param, FOCEI()) for subject in data])
  1 - std(vec(VectorOfArray([icwresi(m, subject, param, vrandeffs) for (subject, vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function ϵshrinkage(m::PumasModel,
                    data::Population,
                    param::NamedTuple,
                    approx::FOCE,
                    randeffs=[empirical_bayes(m, subject, param, FOCE()) for subject in data])
  1 - std(vec(VectorOfArray([icwres(m, subject, param, vrandeffs) for (subject,vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function StatsBase.aic(m::PumasModel,
                       data::Population,
                       param::NamedTuple,
                       approx::LikelihoodApproximation)
  numparam = TransformVariables.dimension(totransform(m.param))
  2*(marginal_nll(m, data, param, approx) + numparam)
end

function StatsBase.bic(m::PumasModel,
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

function StatsBase.predict(model::PumasModel, subject::Subject, param, approx, vrandeffs=empirical_bayes(model, subject, param, approx))
  pred = _predict(model, subject, param, approx, vrandeffs)
  ipred = _ipredict(model, subject, param, approx, vrandeffs)
  SubjectPrediction(pred, ipred, subject, approx)
end

function StatsBase.predict(fpm::FittedPumasModel, approx=fpm.approx; nsim=nothing, timegrid=false, newdata=false, useEBEs=true)
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

  df[!,:pred] .= vcat((pred.pred for pred in vpred)...)
  df[!,:ipred] .= vcat((pred.ipred for pred in vpred)...)
  df[!,:pred_approx] .= vcat((fill(pred.approx, length(pred.subject.time)) for pred in vpred)...)

  df
end

struct SubjectEBES{T1, T2, T3}
  ebes::T1
  subject::T2
  approx::T3
end
function empirical_bayes(fpm::FittedPumasModel, approx=fpm.approx)
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
    df[!,Symbol("ebe_$i")] .= vcat((fill(ebes.ebes[i], length(ebes.subject.time)) for ebes in vebes)...)
  end
  df[!,:ebes_approx] .= vcat((fill(ebes.approx, length(ebes.subject.time)) for ebes in vebes)...)

  df
end

struct FittedPumasModelInspection{T1, T2, T3, T4}
  o::T1
  pred::T2
  wres::T3
  ebes::T4
end
StatsBase.predict(i::FittedPumasModelInspection) = i.pred
wresiduals(i::FittedPumasModelInspection) = i.wres
empirical_bayes(i::FittedPumasModelInspection) = i.ebes

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
  FittedPumasModelInspection(fpm, pred, res, ebes)
end
function DataFrames.DataFrame(i::FittedPumasModelInspection; include_covariates=true)
  pred_df = DataFrame(i.pred; include_covariates=include_covariates)
  res_df = select!(select!(DataFrame(i.wres; include_covariates=false), Not(:id)), Not(:time))
  ebes_df = select!(select!(DataFrame(i.ebes; include_covariates=false), Not(:id)), Not(:time))

  df = hcat(pred_df, res_df, ebes_df)
end
