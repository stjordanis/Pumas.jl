@inline function lt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ + C₂) * Δt) / 2
end

@inline function llt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ - C₂) * Δt) / (log(C₁ / C₂))
end

@inline function mllt(C₁::Real, C₂::Real, Δt::Real)
	Δt = t₂	- t₁
	return ((((t₁ * C₁) - (t₂ * C₂)) * Δt) / (log(C₁ / C₂))) - (((C₂ - C₁) * Δt * Δt)/((log(C₁ / C₂))*(log(C₁ / C₂))))
end

function AUC(concentration::Vector{<:Real}, t::Vector{<:Real}, method::Symbol)
	auc = lt(0, concentration[1], t[1])
	n = length(concentration)

	λ, points, maxrsq, outlier = find_lambda(concentration, t)
	if method == :linear
		for i in 2:n
			auc += lt(concentration[i-1], concentration[i], t[i] - t[i-1])
		end
	elseif method == :log_linear
		for i in 2:n
			if concentration[i] ≥ concentration[i-1]
				auc += lt(concentration[i-1], concentration[i], t[i] - t[i-1])
			else
				auc += llt(concentration[i-1], concentration[i], t[i] - t[i-1])
			end
		end
	else
		throw(ArgumentError("Method must either :linear or :log_linear!"))
	end
	return auc + concentration[n] / λ
end

function AUMC(concentration::Vector{<:Real}, t::Vector{<:Real}, method::Symbol)
	aumc = lt(0, concentration[1]*t[1], t[1])
	n = length(concentration)

	λ, points, maxrsq, outlier = find_lambda(concentration, t)
	if method == :linear
		for i in 2:n
			aumc += lt(concentration[i-1]*t[i-1], concentration[i]*t[i], t[i] - t[i-1])
		end
	elseif method == :log_linear
		for i in 2:n
			if concentration[i] ≥ concentration[i-1]
				aumc += lt(concentration[i-1]*t[i-1], concentration[i]*t[i], t[i] - t[i-1])
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
