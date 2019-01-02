using Test
using PuMaS, LinearAlgebra, Optim

data = process_nmtran(example_nmtran_data("sim_data_model1"))

#likelihood tests from NLME.jl 
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1,init=[0.5])
        Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
        Σ ∈ ConstDomain(0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ[1] * exp(η[1])
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

x0 = init_param(mdsl1)

mdsl_f(η,subject) = PuMaS.penalized_conditional_nll(mdsl1,subject, x0, (η=η,))
ηstar = [Optim.optimize(η -> mdsl_f(η,data[i]),zeros(1),BFGS()).minimizer[1] for i in 1:10]
@test ηstar ≈ [-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361] atol = 1e-3

η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(mdsl1,subject,x0,(η=[0.0],),Laplace()), data.subjects)
@test η0_mll ≈ 57.19397077905644

ml = sum(i -> PuMaS.marginal_nll_nonmem(mdsl1,data[i],x0,(η=[ηstar[i]],),Laplace()), 1:10)
@test ml ≈ 56.810343602063618 rtol = 1e-6

function full_ll(θ)
  _x0 = (θ=θ,Ω=[0.04],Σ=0.1)
  ηstar = [Optim.optimize(η -> mdsl_f(η,data[i]),zeros(1),BFGS()).minimizer[1] for i in 1:10]
  sum(i -> PuMaS.marginal_nll_nonmem(mdsl1,data[i],_x0,(η=[ηstar[i]],),Laplace()), 1:10)
end

Optim.optimize(full_ll,[0.5],BFGS())