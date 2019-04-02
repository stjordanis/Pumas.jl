using Test
using PuMaS, LinearAlgebra, Optim

@test_broken begin
data = process_nmtran(example_nmtran_data("sim_data_model1_wang"))

wang_additive = @model begin
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
        dv ~ @. Normal(conc,sqrt(Σ)+eps())
    end
end

param = init_param(wang_additive)

mdsl_f(η,subject) = PuMaS.penalized_conditional_nll(wang_additive,subject, param, (η=η,))
ηstar = [Optim.optimize(η -> wang_f(η,data[i]),zeros(1),BFGS()).minimizer[1] for i in 1:10]
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_additive,subject,param,(η=[0.0],),FO()), data.subjects)
@test η0_mll ≈ 0.026 atol = 1e-3
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_additive,subject,param,(η=[0.0],),FOCE()), data.subjects)
@test η0_mll ≈ -2.059 atol = 1e-3


wang_prop = @model begin
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

param = init_param(wang_prop)

mdsl_f(η,subject) = PuMaS.penalized_conditional_nll(wang_prop,subject, param, (η=η,))
ηstar = [Optim.optimize(η -> wang_f(η,data[i]),zeros(1),BFGS()).minimizer[1] for i in 1:10]
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_prop,subject,param,(η=[0.0],),FO()), data.subjects)
@test η0_mll ≈ 39.213 atol = 1e-3
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_prop,subject,param,(η=[0.0],),FOCE()), data.subjects)
@test η0_mll ≈ 39.207 atol = 1e-3
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_prop,subject,param,(η=[0.0],),FOCEI()), data.subjects)
@test η0_mll ≈ 39.458 atol = 1e-3

wang_exp = @model begin
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

param = init_param(wang_exp)

mdsl_f(η,subject) = PuMaS.penalized_conditional_nll(wang_exp,subject, param, (η=η,))
ηstar = [Optim.optimize(η -> wang_f(η,data[i]),zeros(1),BFGS()).minimizer[1] for i in 1:10]
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_exp,subject,param,(η=[0.0],),FO()), data.subjects)
#@test η0_mll ≈ 
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_exp,subject,param,(η=[0.0],),FOCE()), data.subjects)
#@test η0_mll ≈ 
η0_mll = sum(subject -> PuMaS.marginal_nll_nonmem(wang_exp,subject,param,(η=[0.0],),FOCEI()), data.subjects)
#@test η0_mll ≈ 

end
