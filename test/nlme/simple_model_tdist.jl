using Test
using PuMaS, LinearAlgebra, Optim

data = process_nmtran(example_nmtran_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
tdist = @model begin
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
        dv ~ @. LocationScale(conc, conc*sqrt(Σ)+eps(), TDist(250))
    end
end


x0 = init_param(tdist)

for (i, v) in enumerate([-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361])
  @test PuMaS.rfx_estimate(tdist, data[i], x0, Laplace())[1] ≈ v rtol=1e-3
end

@test_broken PuMaS.marginal_nll_nonmem(tdist,data,x0,[0.0],Laplace()) ≈ 57.19397077905644 rtol=1e-6
@test_broken PuMaS.marginal_nll_nonmem(tdist,data,x0,(η=[0.0],),Laplace()) ≈ 57.19397077905644 rtol=1e-6
@test_broken PuMaS.marginal_nll_nonmem(tdist,data,x0,Laplace()) ≈ 56.810343602063618 rtol=1e-6

function full_ll(θ)
  _x0 = (θ=θ,Ω=fill(0.04,1,1),Σ=0.1)
  PuMaS.marginal_nll_nonmem(tdist,data,_x0,Laplace())
end

Optim.optimize(full_ll,[0.5],BFGS())

ηstar = [-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361]
ml = sum(i -> PuMaS.marginal_nll_nonmem(tdist,data[i],x0,(η=[ηstar[i]],),PuMaS.FOCEI()), 1:10)
@test_broken ml ≈ 56.410938825140313 rtol = 1e-6

ηstar = [-0.1007, 0.0167, -0.0363, -0.0820, 0.1061, 0.0473, -0.1007, -0.0361, -0.0578, -0.0181]
ml = sum(i -> PuMaS.marginal_nll_nonmem(tdist,data[i],x0,(η=[ηstar[i]],),PuMaS.FOCE()), 1:10)
@test_broken ml ≈ 56.476216665029462 rtol = 1e-6
