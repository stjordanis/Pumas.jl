@inline function lt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ + C₂) * Δt) / 2
end

@inline function llt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ - C₂) * Δt) / (log(C₁ / C₂))
end

@inline function mllt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ - C₂) * Δt) / (log(C₁ / C₂))
end

function AUC(concentration::Vector{<:Real}, Δt::Vector{<:Real}, method::Symbol)
	auc = lt(0, concentration[1], Δt[1])
	n = length(concentration)

	λ, points, maxrsq, outlier = find_lambda(concentration, Δt)
	if method == :linear
		for i in 2:n
			auc += lt(concentration[i-1], concentration[i], Δt[i])
		end
	elseif method == :log_linear
		for i in 2:n
			if concentration[i] ≥ concentration[i-1]
				auc += lt(concentration[i-1], concentration[i], Δt[i])
			else
				auc += llt(concentration[i-1], concentration[i], Δt[i])
			end
		end
	end
	return auc + concentration[n] / λ, auc
end

function find_lambda(concentration::Vector{<:Real}, Δt::Vector{<:Real})
	maxrsq = 0.0
	λ = 0.0
	points = 2
	outlier = false
	for i in 2:10
		data = DataFrame(X = Δt[end-i:end], Y = log.(concentration[end-i:end]))
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
