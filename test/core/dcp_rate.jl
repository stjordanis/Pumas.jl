using PuMaS, Test, LinearAlgebra

@testset "Testing dose control parameters" begin

choose_covariates() = (isPM = rand(["yes", "no"]),
                       Wt = rand(55:80))

function generate_population(events,nsubs=24)
  pop = Population(map(i -> Subject(id=i,evs=events,cvs=choose_covariates()),1:nsubs))
  return pop
end

m_diffeq = @model begin
    @param   begin
        θ ∈ VectorDomain(7, lower=zeros(7), init=ones(7))
        Ω ∈ PSDDomain(2)
        σ_prop ∈ RealDomain(init=0.1)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = θ[1]
        TVCL = isPM == "yes" ? θ[2] : θ[2]*θ[6]
        CL = TVCL*exp(η[1])
        V  = θ[3]*exp(η[2])
        lags = [0,θ[4]]
        bioav = [1,θ[5]]
        duration = [0,θ[7]]
    end

    @covariates isPM Wt

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/V)*Central
    end

    @derived begin
        cp = @. 1000*(Central / V)
    end
end

evm2 = DosageRegimen(100, rate = -2, cmt = 2, ii = 24, addl = 3)
evm216 =  generate_population(evm2)

ev = DosageRegimen(100, rate = 0, cmt = 2, ii = 24, addl = 3)
ev16 =  generate_population(ev)

p = ( θ = [1.5,  #Ka
           1.1,  #CL
           20.0,  #V
           5, # lags2
           0.61, #Bioav
           0.5, # isPM CL
           9 # duration
           ],
      Ω = PDMat(diagm(0 => [0.04,0.04])),
      σ_prop = 0.00
  )


sim = simobs(m_diffeq, ev16, p; abstol=1e-14, reltol=1e-14, parallel_type = PuMaS.Serial)
simm2 = simobs(m_diffeq, evm216, p; abstol=1e-14, reltol=1e-14, parallel_type = PuMaS.Serial)

@test maximum(maximum(s[:cp]) for s in sim.sims) < 1e4
@test maximum(maximum(s[:cp]) for s in simm2.sims) < 1e4

m_error = @model begin
    @param   begin
        θ ∈ VectorDomain(7, lower=zeros(7), init=ones(7))
        Ω ∈ PSDDomain(2)
        σ_prop ∈ RealDomain(init=0.1)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = θ[1]
        TVCL = isPM == "yes" ? θ[2] : θ[2]*θ[6]
        CL = TVCL*exp(η[1])
        V  = θ[3]*exp(η[2])
        lags = [0,θ[4]]
        bioav = [1,θ[5]]
    end

    @covariates isPM Wt

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/V)*Central
    end

    @derived begin
        cp = @. 1000*(Central / V)
    end
end

p_error = (θ = [1.5,  #Ka
                1.1,  #CL
                20.0,  #V
                5, # lags2
                0.61, #Bioav
                0.5, # isPM CL
                ],
           Ω = PDMat(diagm(0 => [0.04,0.04])),
           σ_prop = 0.00
  )

@test_throws ArgumentError simobs(m_error, evm216, p_error, parallel_type = PuMaS.Serial)

end # testset
