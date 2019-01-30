using PuMaS
praz21 = process_nmtran(example_nmtran_data("event_data/praz21"),[:RACE,:HT,:AGE,:HCTZ])
iov_laplace = @model begin
    @param begin
        θ ∈ VectorDomain(8, init=[0.4,0.3,23,1.9,-4.8,103,-27,1.4])
        Ω ∈ PSDDomain(2)
        σ_prop ∈ RealDomain(init=0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        OCC2 = 1 - OCC
        R = RACE >= 2 ? 1 : 0
        TMP = (HT-60)*θ[1]

        TMP2 = (AGE-60)*θ[2]

        TVCL = θ[3] + TMP-TMP2
        TVCL = TVCL + θ[4]*R + θ[5]*HCTZ

        TVV = θ[6] + HCTZ*θ[7]

        TVKA = θ[8]

        CL = TVCL* exp(η[3]*OCC+η[5]*OCC2+η[1])
        V  = TVV * exp(η[4]*OCC+η[6]*OCC2+η[2])
        KA = TVKA* exp(η[8]*OCC+η[9]*OCC2+η[7])

    end

    @covariates RACE HT AGE HCTZ

    @vars begin
        conc = Central / V
    end

    @dynamics OneCompartmentModel

    @derived begin
        dv ~ @. Normal(conc,sqrt(conc^2*σ_prop))
    end
end
x0 = (θ = [0.4,0.3,23,1.9,-4.8,103,-27,1.4],
      Ω = PDMat(diagm(0 => [5.55,0.515])),
      σ_prop = 0.1
           )
