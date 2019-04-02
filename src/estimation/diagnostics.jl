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
  1 - std(vec(VectorOfArray([icwresi(m, subject, param, vrandeffs) for (subject, vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function ϵshrinkage(m::PuMaSModel,
  data::Population,
  param::NamedTuple,
  approx::FOCE,
  randeffs=[randeffs_estimate(m, subject, param, FOCE()) for subject in data])
  1 - std(vec(VectorOfArray([icwres(m, subject, param, vrandeffs) for (subject,vrandeffs) in zip(data, randeffs)])), corrected = false)
end

function AIC(m::PuMaSModel, data::Population, param::NamedTuple, approx::LikelihoodApproximation)
  numparam = TransformVariables.dimension(totransform(m.param))
  marginal_nll_nonmem(m, data, param, approx) + numparam
end

function BIC(m::PuMaSModel, data::Population, param::NamedTuple, approx::LikelihoodApproximation)
  numparam = TransformVariables.dimension(totransform(m.param))
  marginal_nll_nonmem(m, data, param, approx) + numparam*log(sum(t -> length(t.time), data))
end
