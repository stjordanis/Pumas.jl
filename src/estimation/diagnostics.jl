"""
  npde(model, subject, fixeffs, randeffs, simulations_count)

To calculate the Normalised Prediction Distribution Errors (NPDE).
"""
function npde(m::PuMaSModel, subject::Subject, fixeffs::NamedTuple, randeffs::NamedTuple, nsim::Integer)
  y = subject.observations.dv
  sims = [simobs(m, subject, fixeffs, randeffs).observed.dv for i in 1:nsim]
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
              fixeffs::NamedTuple,
              vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FO()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  Ω = Matrix(fixeffs.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, fixeffs, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv))
end

"""
  cwres(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals (CWRES).
"""
function cwres(m::PuMaSModel,
               subject::Subject,
               fixeffs::NamedTuple,
               vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCE()))
  y = subject.observations.dv
  nl0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),))
  nl , dist  = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  Ω = Matrix(fixeffs.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, fixeffs, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist0.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv) .+ F*vrandeffs)
end

"""
  cwresi(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals with Interaction (CWRESI).
"""

function cwresi(m::PuMaSModel,
                subject::Subject,
                fixeffs::NamedTuple,
                vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCEI()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  Ω = Matrix(fixeffs.Ω)
  F = ForwardDiff.jacobian(s -> mean.(conditional_nll_ext(m, subject, fixeffs, (η=s,))[2].dv), vrandeffs)
  V = Symmetric(F*Ω*F' + Diagonal(var.(dist.dv)))
  return cholesky(V).U'\(y .- mean.(dist.dv) .+ F*vrandeffs)
end

"""
  pred(model, subject, param[, rfx])

To calculate the Population Predictions (PRED).
"""
function pred(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FO()))
  l0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),))
  mean_yi = (mean.(dist0.dv))
  mean_yi
end

"""
  cpred(model, subject, param[, rfx])

To calculate the Conditional Population Predictions (CPRED).
"""
function cpred(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCE()))
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  f = Matrix(VectorOfArray([ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), i), vrandeffs) for i in 1:length(subject.observations.dv)]))
  mean_yi = (mean.(dist.dv)) .- vec(f'*vrandeffs)
  mean_yi
end

"""
  cpredi(model, subject, param[, rfx])

To calculate the Conditional Population Predictions with Interaction (CPREDI).
"""
function cpredi(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCEI()))
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  f = Matrix(VectorOfArray([ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), i), vrandeffs) for i in 1:length(subject.observations.dv)]))
  mean_yi = (mean.(dist.dv)) .- vec(f'*vrandeffs)
  mean_yi
end

"""
  epred(model, subject, param[, rfx], simulations_count)

To calculate the Expected Simulation based Population Predictions.
"""
function epred(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple,nsim)
  sims = []
  for i in 1:nsim
    vals = simobs(m, subject, fixeffs)
    push!(sims, vals.observed.dv)
  end
  mean_yi = [mean(sims[:][i]) for i in 1:length(sims[1])]
  mean_yi
end

"""
  iwres(model, subject, param[, rfx])

To calculate the Individual Weighted Residuals (IWRES).
"""
function iwres(m::PuMaSModel,
               subject::Subject,
               fixeffs::NamedTuple,
               vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FO()))
  y = subject.observations.dv
  nl, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist.dv)
end

"""
  icwres(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals (ICWRES).
"""
function icwres(m::PuMaSModel,
                subject::Subject,
                fixeffs::NamedTuple,
                vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCE()))
  y = subject.observations.dv
  nl0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),))
  nl , dist  = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist0.dv)
end

"""
  icwresi(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals with Interaction (ICWRESI).
"""
function icwresi(m::PuMaSModel,
                 subject::Subject,
                 fixeffs::NamedTuple,
                 vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCEI()))
  y = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  return (y .- mean.(dist.dv)) ./ std.(dist.dv)
end

"""
  eiwres(model, subject, param[, rfx], simulations_count)

To calculate the Expected Simulation based Individual Weighted Residuals (EIWRES).
"""
function eiwres(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, nsim)
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,fixeffs)
  mean_yi = (mean.(dist.dv))
  covm_yi = sqrt(inv((Diagonal(var.(dist.dv)))))
  sims_sum = (covm_yi)*(yi .- mean_yi)
  for i in 2:nsim
    l, dist = conditional_nll_ext(m,subject,fixeffs)
    mean_yi = (mean.(dist.dv))
    covm_yi = sqrt(inv((Diagonal(var.(dist.dv)))))
    sims_sum .+= (covm_yi)*(yi .- mean_yi)
  end
  sims_sum./nsim
end

function ηshrinkage(m::PuMaSModel, data::Population, fixeffs::NamedTuple, approx)
  sd_randeffs = std(Matrix(VectorOfArray([randeffs_estimate(m, subject, fixeffs, approx) for subject in data])),dims=2)
  Ω = (fixeffs.Ω)
  shk = 1 .- (sd_randeffs ./ sqrt.(diag(Ω)))
  shk
end

function ϵshrinkage(m::PuMaSModel, data::Population, fixeffs::NamedTuple)
  1 - std(vec([iwres(m, subject, fixeffs) for subject in data]))[1]
end
