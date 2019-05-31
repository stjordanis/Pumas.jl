# For Vitro data

# Emax model
# p = [D_INF, γ, TD50, time_lag(optional)]
function emax(t::Number, p)
  if length(p) == 4
    t = t - p[4]
    t = t <= zero(t) ? zero(t) : t
  end
  return ((p[1] * (t ^ p[2])) / (p[3]^p[2] + t^p[2]))
end

# Emax model without γ
# p = [D_INF, TD50, time_lag(optional)]
function emax_ng(t::Number, p)
  if length(p) == 3
    t = t - p[3]
    t = t <= zero(t) ? zero(t) : t
  end
  return (p[1] * t) / (p[2] + t)
end

# Weibull Model
# p = [D_INF, Mean_dissolution_time, γ, time_lag(optional)]
function weibull(t::Number, p)
  if length(p) == 4
    t = t - p[4]
    t = t <= zero(t) ? zero(t) : t
  end
  return p[1] * (1.0 - exp(-(((t)/p[2])^p[3])))
end

# Double Weibull model
# p = [F1, F_INF, MDT1, B1, MDT2, B2, time_lag(optional)]
# MDT = Mean_dissolution_time, B1 and B2 are hill coefficients
function double_weibull(t::Number, p)
  if length(p) == 7
    t = t - p[7]
    t = t <= zero(t) ? zero(t) : t
  end
  return (p[1] * p[2] * (1.0 - exp(-(t/p[3])^p[4]))) + ((1.0-p[1]) * p[2] * (1.0 - exp(-(t/p[5])^p[6])))
end

# Makoid-Banakar model
# p = [D_INF, Mean_dissolution_time, γ, time_lag(optional)]
function makoid(t::Number, p)
  if length(p) == 4
    t = t - p[4]
    t = t <= zero(t) ? zero(t) : t
  end
  res = zero(t)
  if t <= p[2]
    res = p[1] * ((t/p[2])^p[3]) * exp(p[3]*(1.0-t/p[2]))
  else
    res = p[1] * one(t)
  end
  res
end

emax(t::AbstractVector, p) = [emax(x, p) for x in t]
emax_ng(t::AbstractVector, p) = [emax_ng(x, p) for x in t]
weibull(t::AbstractVector, p) = [weibull(x, p) for x in t]
double_weibull(t::AbstractVector, p) = [double_weibull(x, p) for x in t]
makoid(t::AbstractVector, p) = [makoid(x, p) for x in t]

# For Vivo data
# bateman function
function bateman(t::Number, p)
  ka = p[1]; kel = p[2]; V = p[3]
  (ka/(V*(ka - kel))) * (exp(-kel*t) - exp(-ka*t))
end

bateman(t::AbstractVector, p) = [bateman(x, p) for x in t]
