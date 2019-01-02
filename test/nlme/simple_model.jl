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

for (i, v) in enumerate([-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361])
  @test PuMaS.rfx_estimate(mdsl1, data[i], x0, Laplace())[1] ≈ v rtol=1e-3
end

@test PuMaS.marginal_nll_nonmem(mdsl1,data,x0,[0.0],Laplace()) ≈ 57.19397077905644 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1,data,x0,(η=[0.0],),Laplace()) ≈ 57.19397077905644 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1,data,x0,Laplace()) ≈ 56.810343602063618 rtol=1e-6

function full_ll(θ)
  _x0 = (θ=θ,Ω=[0.04],Σ=0.1)
  PuMaS.marginal_nll_nonmem(mdsl1,data,_x0,Laplace())
end

Optim.optimize(full_ll,[0.5],BFGS())
