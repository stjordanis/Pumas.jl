function npde(m::PuMaSModel,subject::Subject, x0::NamedTuple,nsim)
  yi = subject.observations.dv
  sims = []
  for i in 1:nsim
    vals = simobs(m, subject, x0)
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

function wres(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FO()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  mean_yi = (mean.(dist0.dv))
  Ω = cov(m.random(x0).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations.dv)]
  var_yi = sqrt(inv((Diagonal(var.(dist0.dv))) + (f*Ω*f')))
  (var_yi)*(yi .- mean_yi)
end

function cwres(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCE()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  l, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  Ω = cov(m.random(x0).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations.dv)]
  var_yi = sqrt(inv((Diagonal(var.(dist0.dv))) + (f*Ω*f')))
  mean_yi = (mean.(dist.dv)) .- vec(f*vy0')
  (var_yi)*(yi .- mean_yi)
end

function cwresi(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCEI()))
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  Ω = cov(m.random(x0).params.η)
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations.dv)]
  var_yi = sqrt(inv((Diagonal(var.(dist.dv))) + (f*Ω*f')))
  mean_yi = (mean.(dist.dv)) .- vec(f*vy0')
  (var_yi)*(yi .- mean_yi)
end

function pred(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FO()))
  l0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  mean_yi = (mean.(dist0.dv))
  mean_yi
end

function cpred(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCE()))
  l, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations.dv)]
  mean_yi = (mean.(dist.dv)) .- vec(f*vy0')
  mean_yi
end

function cpredi(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCEI()))
  l, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  f = [ForwardDiff.gradient(s -> _mean(m, subject, x0, (η=s,), i), vy0)[1] for i in 1:length(subject.observations.dv)]
  mean_yi = (mean.(dist.dv)) .- vec(f*vy0')
  mean_yi
end

function epred(m::PuMaSModel,subject::Subject, x0::NamedTuple,nsim)
  sims = []
  for i in 1:nsim
    vals = simobs(m, subject, x0)
    push!(sims, vals.observed.dv)
  end
  mean_yi = [mean(sims[:][i]) for i in 1:length(sims[1])]
  mean_yi
end

function iwres(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FO()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  mean_yi = (mean.(dist0.dv))
  sqrt(inv((Diagonal(var.(dist0.dv)))))*(yi .- mean_yi)
end

function icwres(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCE()))
  yi = subject.observations.dv
  l0, dist0 = conditional_nll_ext(m,subject,x0, (η=zero(vy0),))
  l, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  mean_yi = (mean.(dist.dv))
  sqrt(inv((Diagonal(var.(dist0.dv)))))*(yi .- mean_yi)
end

function icwresi(m::PuMaSModel,subject::Subject, x0::NamedTuple, vy0::AbstractVector=rfx_estimate(m, subject, x0, FOCEI()))
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,x0, (η=vy0,))
  mean_yi = (mean.(dist.dv))
  sqrt(inv((Diagonal(var.(dist.dv)))))*(yi .- mean_yi)
end

function eiwres(m::PuMaSModel,subject::Subject, x0::NamedTuple, nsim)
  yi = subject.observations.dv
  l, dist = conditional_nll_ext(m,subject,x0)
  mean_yi = (mean.(dist.dv))
  covm_yi = sqrt(inv((Diagonal(var.(dist.dv)))))
  sims_sum = (covm_yi)*(yi .- mean_yi)
  for i in 2:nsim
    l, dist = conditional_nll_ext(m,subject,x0)
    mean_yi = (mean.(dist.dv))
    covm_yi = sqrt(inv((Diagonal(var.(dist.dv)))))
    sims_sum .+= (covm_yi)*(yi .- mean_yi)
  end
  sims_sum./nsim
end
