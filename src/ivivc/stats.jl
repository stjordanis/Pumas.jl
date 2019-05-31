loglikelihood(obj::Ivivc) = -nobs(obj)//2 * log(rss(obj)/nobs(obj))
nullloglikelihood(obj::Ivivc) = error("not implemented yet!")

dof(obj::Ivivc) = length(obj.pmin)
nobs(obj::Ivivc) = length(obj.time)
mean(x::AbstractVector) = sum(x)/length(x)
deviance(obj::Ivivc) = sum((obj.conc .- mean(obj.conc)) .^ 2)
mss(obj::Ivivc) = sum((obj(obj.time) .- mean(obj.conc)) .^ 2)
rss(obj::Ivivc)  = sum((obj(obj.time) .- obj.conc) .^ 2)

aic(obj::Ivivc) = -2loglikelihood(obj) + 2dof(obj)

function aicc(obj::Ivivc)
  k = dof(obj)
  n = nobs(obj)
  -2loglikelihood(obj) + 2k + 2k*(k+1)/(n-k-1)
end

bic(obj::Ivivc) = -2loglikelihood(obj) + dof(obj)*log(nobs(obj))

r2(obj::Ivivc) = mss(obj) / deviance(obj)

function r2(obj::Ivivc, variant::Symbol)
  ll = loglikelihood(obj)
  ll0 = nullloglikelihood(obj)
  if variant == :McFadden
    1 - ll/ll0
  elseif variant == :CoxSnell
    1 - exp(2 * (ll0 - ll) / nobs(obj))
  elseif variant == :Nagelkerke
    (1 - exp(2 * (ll0 - ll) / nobs(obj))) / (1 - exp(2 * ll0 / nobs(obj)))
  else
    error("variant must be one of :McFadden, :CoxSnell or :Nagelkerke")
  end
end

function summary(obj::Ivivc)
  error("not implemented yet!")
end
