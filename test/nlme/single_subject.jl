using Test
using Pumas, LinearAlgebra

@testset "Single subject" begin
data = read_pumas(example_nmtran_data("sim_data_model1"))
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1, init=[0.5])
        Ω ∈ ConstDomain(Diagonal([0.04]))
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

param = init_param(mdsl1)

for approx in (Pumas.FO, Pumas.FOI, Pumas.FOCE, Pumas.FOCEI, Pumas.Laplace, Pumas.LaplaceI, Pumas.HCubeQuad)
    @test_throws ArgumentError fit(mdsl1, data[1], param, approx())
end

@test_throws ErrorException fit(mdsl1, data[1], param)

end
