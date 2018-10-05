@inline function lt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ + C₂) * Δt) / 2
end

@inline function llt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ - C₂) * Δt) / (log(C₁ / C₂))
end

@inline function mllt(C₁::Real, C₂::Real, Δt::Real)
	return ((C₁ - C₂) * Δt) / (log(C₁ / C₂))
end

function AUC(concentration::Vector{Real}, Δt::Vector{Real}, method::Symbol)
	auc = lt(0, concentration[1], Δt[1])
	n =length(concentration)

	λ = polyfit(concentration[n-3:n], Δt[n-3:n], 1).a[2]
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
	return auc + concentration[n] / λ
end
