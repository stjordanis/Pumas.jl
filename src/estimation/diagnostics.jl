import StatsBase: predict
"""
  npde(model, subject, param, randeffs, simulations_count)

To calculate the Normalised Prediction Distribution Errors (NPDE).
"""
function npde(m::PuMaSModel, subject::Subject, param::NamedTuple, randeffs::NamedTuple, nsim::Integer)
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

"""

  residuals(fpm::FittedPuMasModel)
Calculates a vector of weighted residual according to the approximation method used in
the fitting procedure whose result is saved in `fpm`.
"""
function residuals(fpm, approx=fpm.approx)
  n_obs = length(fpm.data)
  [residuals(approx, fpm, subject_index) for subject_index = 1:n_obs]
end
"""

iresiduals(fpm::FittedPuMasModel)
Calculates a vector of weighted residual according to the approximation method used in
the fitting procedure whose result is saved in `fpm`.
"""
function iresiduals(fpm, approx=fpm.approx)
  n_obs = length(fpm.data)
  [iresiduals(approx, fpm, subject_index) for subject_index = 1:n_obs]
end

"""

  wres(model, subject, param[, rfx])

To calculate the Weighted Residuals (WRES).
"""
function wres(m::PuMaSModel,
              subject::Subject,
              param::NamedTuple,
              vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FO()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv))
end
function residuals(approx::FO, fpm, subject_index)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  randeffs = fpm.vrandeffs[subject_index]
  wres(model, subject, param, randeffs)
end

"""
  cwres(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals (CWRES).
"""
function cwres(m::PuMaSModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FOCE()))
  y = subject.observations.dv
  nl0, dist0 = conditional_nll_ext(m,subject,param, (η=zero(vrandeffs),))
  nl , dist  = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist0.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv) .+ F*vrandeffs)
end
function residuals(approx::FOCE, fpm, subject_index)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  randeffs = fpm.vrandeffs[subject_index]
  cwres(model, subject, param, randeffs)
end
"""
  cwresi(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals with Interaction (CWRESI).
"""

function cwresi(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FOCEI()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m, subject, param, (η=vrandeffs,))
  Ω = Matrix(param.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv) .+ F*vrandeffs)
end
function residuals(approx::FOCEI, fpm, subject_index)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  vrandeffs = fpm.vrandeffs[subject_index]
  cwresi(model, subject, param, vrandeffs)
end

StatsBase.predict(fpm::FittedPuMaSModel) = predict(fpm, fpm.approx)
function StatsBase.predict(fpm::FittedPuMaSModel, approx)
  n_obs = length(fpm.data)
  [predict(fpm, fpm.approx, subject_index) for subject_index = 1:n_obs]
end

"""
  pred(model, subject, param[, rfx])

To calculate the Population Predictions (PRED).
"""
function pred(m::PuMaSModel,
              subject::Subject,
              param::NamedTuple,
              vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FO()))
  nl, dist = conditional_nll_ext(m, subject, param, (η=vrandeffs,))
  return mean.(dist.dv)
end
function predict(fpm, approx::FO, i)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  vrandeffs = fpm.vrandeffs[subject_index]
  pred(m, subject, param, vrandeffs)
end


"""
  cpred(model, subject, param[, rfx])

To calculate the Conditional Population Predictions (CPRED).
"""
function cpred(m::PuMaSModel,
               subject::Subject,
               param::NamedTuple,
               vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FOCE()))
  nl, dist = conditional_nll_ext(m, subject,param, (η=vrandeffs,))
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  return mean.(dist.dv) .- F*vrandeffs
end
function predict(fpm, approx::FOCE, i)
  model = fpm.model
  subject = fpm.data.subjects[i]
  param = fpm.param
  vrandeffs = fpm.vrandeffs[i]
  cpred(model, subject, param, vrandeffs)
end

"""
  cpredi(model, subject, param[, rfx])

To calculate the Conditional Population Predictions with Interaction (CPREDI).
"""
function cpredi(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FOCEI()))
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, param, (η=s,))[2].dv), vrandeffs)
  return mean.(dist.dv) .- F*vrandeffs
end
function predict(fpm, approx::FOCEI, subject_index)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  vrandeffs = fpm.vrandeffs[subject_index]
  cpredi(model, subject, param, vrandeffs)
end

function epredict(fpm::FittedPuMaSModel, nsim::Integer)
  n_obs = length(fpm.data)
  model = fpm.model
  subjects = fpm.data.subjects
  param = fpm.param
  vrandeffs = fpm.vrandeffs
  [epred(fpm.model, subjects[subject_index], param, (η=vrandeffs[subject_index],), nsim) for subject_index = 1:n_obs]
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
               vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FO()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist.dv)
end
function iresiduals(approx::FO, fpm, subject_index)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  randeffs = fpm.vrandeffs[subject_index]
  iwres(model, subject, param, randeffs)
end
"""
  icwres(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals (ICWRES).
"""
function icwres(m::PuMaSModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FOCE()))
  y = subject.observations.dv
  nl0, dist0 = conditional_nll_ext(m,subject,param, (η=zero(vrandeffs),))
  nl , dist  = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist0.dv)
end
function iresiduals(approx::FOCE, fpm, subject_index)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  randeffs = fpm.vrandeffs[subject_index]
  icwres(model, subject, param, randeffs)
end
"""
  icwresi(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals with Interaction (ICWRESI).
"""
function icwresi(m::PuMaSModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffs::AbstractVector=randeffs_estimate(m, subject, param, FOCEI()))
  y = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,param, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist.dv)
end
function iresiduals(approx::FOCEI, fpm, subject_index)
  model = fpm.model
  subject = fpm.data.subjects[subject_index]
  param = fpm.param
  randeffs = fpm.vrandeffs[subject_index]
  icwresi(model, subject, param, randeffs)
end
"""
  eiwres(model, subject, param[, rfx], simulations_count)

To calculate the Expected Simulation based Individual Weighted Residuals (EIWRES).
"""
function eiwres(m::PuMaSModel,subject::Subject, param::NamedTuple, nsim)
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,param)
  mean_yi = (mean.(dist.dv))
  covm_yi = sqrt(inv((Diagonal(var.(dist.dv)))))
  sims_sum = (covm_yi)*(yi .- mean_yi)
  for i in 2:nsim
    l, dist = conditional_nll_ext(m,subject,param)
    mean_yi = (mean.(dist.dv))
    covm_yi = sqrt(inv((Diagonal(var.(dist.dv)))))
    sims_sum .+= (covm_yi)*(yi .- mean_yi)
  end
  sims_sum./nsim
end

function ηshrinkage(m::PuMaSModel, data::Population, param::NamedTuple, approx)
  sd_randeffs = std([randeffs_estimate(m, subject, param, approx) for subject in data])
  Ω = Matrix(param.Ω)
  return  1 .- sd_randeffs ./ sqrt.(diag(Ω))
end

function ϵshrinkage(m::PuMaSModel,
                    data::Population,
                    param::NamedTuple,
                    approx::FOCEI,
                    randeffs=[randeffs_estimate(m, subject, param, FOCEI()) for subject in data])
  1 - std(vec(VectorOfArray([icwresi(m, subject, param, vrandeffs) for (subject,vrandeffs) in zip(data,randeffs)])),corrected = false)
end

function ϵshrinkage(m::PuMaSModel,
  data::Population,
  param::NamedTuple,
  approx::FOCE,
  randeffs=[randeffs_estimate(m, subject, param, FOCE()) for subject in data])
  1 - std(vec(VectorOfArray([icwres(m, subject, param, vrandeffs) for (subject,vrandeffs) in zip(data,randeffs)])),corrected = false)
end
