"""
    lt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the curve (AUC) in an interval by linear trapezoidal rule.
"""
@inline function lt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
  return ((C₁ + C₂) * Δt) / 2
end

"""
    llt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the curve (AUC) in an interval by log-linear trapezoidal
rule.
"""
@inline function llt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
  return ((C₁ - C₂) * Δt) / (log(C₁ / C₂))
end

"""
    mlt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the first moment of the concentration (AUMC) in an interval
by linear trapezoidal rule.
"""
@inline function mlt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  return lt(C₁*t₁, C₂*t₂, t₁, t₂)
end

"""
    mllt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)

Compute area under the first moment of the concentration (AUMC) in an interval
by log-linear trapezoidal rule.
"""
@inline function mllt(C₁::Real, C₂::Real, t₁::Real, t₂::Real)
  Δt = t₂ - t₁
  return ((((t₁ * C₁) - (t₂ * C₂)) * Δt) / (log(C₁ / C₂))) - (((C₂ - C₁) * Δt * Δt)/((log(C₁ / C₂))*(log(C₁ / C₂))))
end

"""
    AUC(concentration::Vector{<:Real}, t::Vector{<:Real}, method::Symbol)

Compute area under the curve (AUC) by linear trapezoidal rule (method =
:linear) or by log-linear trapezoidal rule (method = :log_linear).
"""
function AUC(concentration::Vector{<:Real}, t::Vector{<:Real}, method::Symbol)
  auc = lt(zero(concentration[1]), concentration[1], zero(t[1]), t[1])
  n = length(concentration)

  λ, points, maxrsq, outlier = find_lambda(concentration, t)
  if method == :linear
    for i in 2:n
      auc += lt(concentration[i-1], concentration[i], t[i-1], t[i])
    end
  elseif method == :log_linear
    for i in 2:n
      if concentration[i] ≥ concentration[i-1]
        auc += lt(concentration[i-1], concentration[i], t[i-1], t[i])
      else
        auc += llt(concentration[i-1], concentration[i], t[i-1], t[i])
      end
    end
  else
    throw(ArgumentError("Method must either :linear or :log_linear!"))
  end
  return auc + concentration[n] / λ
end

"""
    AUMC(concentration::Vector{<:Real}, t::Vector{<:Real}, method::Symbol)

Compute area under the first moment of the concentration (AUMC) by linear
trapezoidal rule (method = :linear) or by log-linear trapezoidal rule (method =
:log_linear).
"""
function AUMC(concentration::Vector{<:Real}, t::Vector{<:Real}, method::Symbol)
  aumc = mlt(zero(concentration[1]), concentration[1], zero(t[1]), t[1])
  n = length(concentration)

  λ, points, maxrsq, outlier = find_lambda(concentration, t)
  if method == :linear
    for i in 2:n
      aumc += mlt(concentration[i-1], concentration[i], t[i-1], t[i])
    end
  elseif method == :log_linear
    for i in 2:n
      if concentration[i] ≥ concentration[i-1]
        aumc += mlt(concentration[i-1], concentration[i], t[i-1], t[i])
      else
        aumc += mllt(concentration[i-1], concentration[i], t[i-1], t[i])
      end
    end
  else
    throw(ArgumentError("Method must either :linear or :log_linear!"))
  end
  return aumc + ((concentration[n] * t[n])/ λ) + (concentration[n]/(λ * λ))
end

function find_lambda(concentration::Vector{<:Real}, t::Vector{<:Real})
  maxrsq = 0.0
  λ = 0.0
  points = 2
  outlier = false
  for i in 2:10
    data = DataFrame(X = t[end-i:end], Y = log.(concentration[end-i:end]))
    model = lm(@formula(Y ~ X), data)
    rsq = r²(model)
    if rsq > maxrsq
      maxrsq = rsq
      λ = coef(model)[2]
      points = i+1
    end
  end
  if λ ≥ 0
    outlier = true
  end
  -λ, points, maxrsq, outlier
end
