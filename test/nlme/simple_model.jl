using Test
using PuMaS, LinearAlgebra, Optim

data = process_nmtran(example_nmtran_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1,init=[0.5])
        Ω ∈ PDiagDomain(PDiagMat(fill(0.04, 1)))
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

for (ηstar, dt) in zip([-0.1007, 0.0167, -0.0363, -0.0820, 0.1061, 0.0473, -0.1007, -0.0361, -0.0578, -0.0181], data)
    @test PuMaS.rfx_estimate(mdsl1, dt, x0, PuMaS.Laplace())[1] ≈ ηstar rtol=1e-2
end
for (ηstar, dt) in zip([-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361], data)
    @test PuMaS.rfx_estimate(mdsl1, dt, x0, PuMaS.LaplaceI())[1] ≈ ηstar rtol=1e-3
end

@test PuMaS.marginal_nll_nonmem(mdsl1, data, x0, PuMaS.FOCEI())    ≈ 56.410938825140313 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, x0, PuMaS.FOCE())     ≈ 56.476216665029462 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, x0, PuMaS.FO())       ≈ 56.474912258255571 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, x0, PuMaS.Laplace())  ≈ 56.613069180382027 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, x0, PuMaS.LaplaceI()) ≈ 56.810343602063618 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, x0, [0.0],PuMaS.LaplaceI()) ≈ 57.19397077905644 rtol=1e-6
@test PuMaS.marginal_nll_nonmem(mdsl1, data, x0, (η=[0.0],),PuMaS.LaplaceI()) ≈ 57.19397077905644 rtol=1e-6

@test fit(mdsl1, data, init_param(mdsl1), PuMaS.LaplaceI(), optimmethod=BFGS(), optimautodiff=:finite) isa PuMaS.FittedPKPDModel
@test fit(mdsl1, data, init_param(mdsl1), PuMaS.LaplaceI(), optimmethod=Newton(), optimautodiff=:finite) isa PuMaS.FittedPKPDModel
@test fit(mdsl1, data, init_param(mdsl1), PuMaS.LaplaceI(), optimmethod=BFGS(), optimautodiff=:forward) isa PuMaS.FittedPKPDModel
@test fit(mdsl1, data, init_param(mdsl1), PuMaS.LaplaceI(), optimmethod=Newton(), optimautodiff=:forward) isa PuMaS.FittedPKPDModel

println([npde(mdsl1,data[i],x0,10000) for i in 1:10]) 
println([cpred(mdsl1,data[i],x0) for i in 1:10]) 
println([cpredi(mdsl1,data[i],x0) for i in 1:10]) 
for (sub_pred, dt) in zip([[10.0000000, 6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660], [10.0000000,6.06530660]] , data)
    @test pred(mdsl1,dt,x0) ≈ sub_pred rtol=1e-6
end

for (sub_wres, dt) in zip([ [0.180566054, 1.74797817]  ,[-1.35845124,-0.274456699],[0.310535666, 0.611240923] ,[0.394652252, 1.41153536]  ,[0.607473539, -1.68539881] ,[0.858874613,-0.769228457],[0.245708974,1.74827643]  ,[-0.169086986,0.608506828],[  -1.38172560,0.984121759] ,[0.905043866,0.302785305]  ] , data)
    @test wres(mdsl1,dt,x0) ≈ sub_wres rtol=1e-6
end

for (sub_cwres, dt) in zip([ [0.180566054, 1.75204867]  ,[-1.35845124,-0.274353057],[0.310535666, 0.611748221] ,[0.394652252, 1.41420526]  ,[0.607473539, -1.68142230] ,[0.858874613,-0.768408883],[0.245708974,1.75234831]  ,[-0.169086986,0.609009620],[  -1.38172560,0.985428904] ,[0.905043866,0.302910385]  ] , data)
    @test cwres(mdsl1,dt,x0) ≈ sub_cwres rtol=1e-6
end

for (sub_cwresi, dt) in zip([ [0.180566054, 1.6665779]  ,[-1.35845124, -0.278938663],[0.310535666, 0.605059261] ,[0.394652252,1.36101861  ]  ,[0.607473539,-1.74177468  ] ,[  0.858874613,-0.789814478],[0.245708974,1.6668457]  ,[ -0.169086986, 0.602404841],[ -1.3817256, 0.962485383] ,[0.905043866, 0.302554671]  ] , data)
   @test cwresi(mdsl1,dt,x0) ≈ sub_cwresi rtol=1e-6
end   