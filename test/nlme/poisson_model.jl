using PuMaS, CSV,Test

# avoid rand issue for now
using StatsFuns, ForwardDiff
StatsFuns.RFunctions.poisrand(x::ForwardDiff.Dual) = StatsFuns.RFunctions.poisrand(ForwardDiff.value(x))

df = process_nmtran(example_nmtran_data("sim_poisson"),[:dose])

poisson_model = @model begin
    @param begin
        θ ∈ VectorDomain(2,init=[1.02930E+00,4.51854E-01])
        Ω ∈ PSDDomain(1.22009E-01)
    end

    @random begin
        η ~ Normal(Ω)
    end

    @pre begin
        baseline= θ[1]*exp(η[1])
        d50 = θ[2]
        λ=baseline*(1-dose/(dose+d50))
    end

    @covariates dose

    @derived begin
        dv ~ @. Poisson(λ)
    end
end


x0 = init_param(poisson_model)
y0 = init_random(poisson_model, x0)

@test solve(poisson_model,df,x0,y0) === nothing
@test simobs(poisson_model,df,x0,y0) != nothing

res = simobs(poisson_model,df,x0,y0)

@test_broken PuMaS.marginal_nll_nonmem(poisson_model,df,x0,y0,Laplace()) ≈ 3809.805993678949562 rtol=1e-6
