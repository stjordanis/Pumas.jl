using Test
using PuMaS

data = process_nmtran(example_nmtran_data("sim_data_model1"))

mdsl = @model begin
    @param begin
        θ ∈ ConstDomain(0.5)
        Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        Σ ∈ ConstDomain(0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ * exp(η[1])
        V  = 1.0
    end

    @vars begin
        conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
        dv ~ @. Normal(conc,conc*sqrt(Σ)+eps())
    end
end

x0 = init_param(mdsl)

using Optim
f(η,subject) = PuMaS.penalized_conditional_nll(mdsl,subject, x0, (η=η,))
ηstar = [Optim.optimize(η -> f(η,data[i]),zeros(1),BFGS()).minimizer[1] for i in 1:10]
@test ηstar ≈ [-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361] atol = 1e-3

η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(mdsl,subject,x0,(η=[0.0],),Laplace()), data.subjects)
@test η0_mll ≈ 57.19397077905644

ml = sum(i -> PuMaS.marginal_nll_nonmem(mdsl,data[i],x0,(η=[ηstar[i]],),Laplace()), 1:10)
@test ml ≈ 56.810343602063618 rtol = 1e-6
