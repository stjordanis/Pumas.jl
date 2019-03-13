"""
  npde(model, subject, param[, rfx], simulations_count)

To calculate the Normalised Prediction Distribution Errors (NPDE).
"""
function npde(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple,nsim)
  yi = subject.observations.dv
  sims = []
  for i in 1:nsim
    vals = simobs(m, subject, fixeffs)
    push!(sims, vals.observed.dv)
  end
  mean_yi = [mean(sims[:][i]) for i in 1:length(sims[1])]
  covm_yi = cov(sims)
  covm_yi = sqrt(inv(covm_yi))
  yi_decorr = (covm_yi)*(yi .- mean_yi)
  phi = []
  for i in 1:nsim
    yi_i = sims[i]
    yi_decorr_i = (covm_yi)*(yi_i .- mean_yi)
    push!(phi,[yi_decorr_i[j]>=yi_decorr[j] ? 0 : 1 for j in 1:length(yi_decorr_i)])
  end
  phi = sum(phi)/nsim
  [quantile(Normal(),phi[i]) for i in 1:length(subject.observations.dv)]
end

"""
  wres(model, subject, param[, rfx])

To calculate the Weighted Residuals (WRES).
"""
function wres(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FO()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),))
  mean_yi = (mean.(dist0.dv))
  Ω = cov(m.random(fixeffs).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), i), vrandeffs)[1] for i in 1:length(subject.observations.dv)]
  var_yi = sqrt(inv((Diagonal(var.(dist0.dv))) + (f*Ω*f')))
  (var_yi)*(yi .- mean_yi)
end

"""
  cwres(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals (CWRES).
"""
function cwres(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCE()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),))
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  Ω = cov(m.random(fixeffs).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), i), vrandeffs)[1] for i in 1:length(subject.observations.dv)]
  var_yi = sqrt(inv((Diagonal(var.(dist0.dv))) + (f*Ω*f')))
  mean_yi = (mean.(dist.dv)) .- vec(f*vrandeffs')
  (var_yi)*(yi .- mean_yi)
end

"""
  cwresi(model, subject, param[, rfx])

To calculate the Conditional Weighted Residuals with Interaction (CWRESI).
"""

function cwresi(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCEI()))
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  Ω = cov(m.random(fixeffs).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), i), vrandeffs)[1] for i in 1:length(subject.observations.dv)]
  var_yi = sqrt(inv((Diagonal(var.(dist.dv))) + (f*Ω*f')))
  mean_yi = (mean.(dist.dv)) .- vec(f*vrandeffs')
  (var_yi)*(yi .- mean_yi)
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
  f = [ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), i), vrandeffs)[1] for i in 1:length(subject.observations.dv)]
  mean_yi = (mean.(dist.dv)) .- vec(f*vrandeffs')
  mean_yi
end

"""
  cpredi(model, subject, param[, rfx])

To calculate the Conditional Population Predictions with Interaction (CPREDI).
"""
function cpredi(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCEI()))
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  f = [ForwardDiff.gradient(s -> _mean(m, subject, fixeffs, (η=s,), i), vrandeffs)[1] for i in 1:length(subject.observations.dv)]
  mean_yi = (mean.(dist.dv)) .- vec(f*vrandeffs')
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
function iwres(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FO()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),))
  mean_yi = (mean.(dist0.dv))
  sqrt(inv((Diagonal(var.(dist0.dv)))))*(yi .- mean_yi)
end

"""
  icwres(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals (ICWRES).
"""
function icwres(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCE()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,fixeffs, (η=zero(vrandeffs),))
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  mean_yi = (mean.(dist.dv))
  sqrt(inv((Diagonal(var.(dist0.dv)))))*(yi .- mean_yi)
end

"""
  icwresi(model, subject, param[, rfx])

To calculate the Individual Conditional Weighted Residuals with Interaction (ICWRESI).
"""
function icwresi(m::PuMaSModel,subject::Subject, fixeffs::NamedTuple, vrandeffs::AbstractVector=randeffs_estimate(m, subject, fixeffs, FOCEI()))
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,fixeffs, (η=vrandeffs,))
  mean_yi = (mean.(dist.dv))
  sqrt(inv((Diagonal(var.(dist.dv)))))*(yi .- mean_yi)
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
  sd_vy0 = std([randeffs_estimate(m, subject, fixeffs, approx) for subject in data])
  Ω = (fixeffs.Ω)
  shk = 1 .- (sd_vy0 ./ sqrt.(diag(Ω)))
  shk
end

function ϵshrinkage(m::PuMaSModel, data::Population, x0::NamedTuple)
  1 - std(vec([iwres(m, subject, x0) for subject in data]))[1]
end
