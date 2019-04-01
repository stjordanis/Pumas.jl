using PuMaS, Test

@testset "Logistic regression example" begin

    data = process_nmtran(joinpath(dirname(pathof(PuMaS)), "..", "examples", "pain_remed.csv"),
                          [:arm, :dose, :conc, :painord,:remed];
                          time=:time, event_data=false)

    mdsl = @model begin
        @param begin
            θ ∈ VectorDomain(2,init=[0.001, 0.0001])
            Ω ∈ PSDDomain(1)
        end

        @random begin
            η ~ MvNormal(Ω)
        end

        @covariates arm dose

        @pre begin
            rx = dose > 0 ? 1 : 0
            INT = θ[1]
            LNPTRT = θ[2]*rx
            LOGIT = INT + LNPTRT + η[1]
        end

        @derived begin
            dv ~ @. Bernoulli(logistic(LOGIT))
        end

    end

    param = (θ=[0.01, 0.001], Ω=PuMaS.PDMat(fill(1.0, 1, 1)))

    @testset "testing with $approx approximation" for
        approx in (PuMaS.FO(), PuMaS.FOCE(), PuMaS.FOCEI(), PuMaS.Laplace(), PuMaS.LaplaceI())

        if approx == PuMaS.LaplaceI()
            @test PuMaS.marginal_nll(mdsl, data, param, approx) isa Number
        else
            @test_broken PuMaS.marginal_nll(mdsl, data, param, approx) isa Number
        end
    end

end
