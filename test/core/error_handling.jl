using Test, Pumas, LinearAlgebra

#Creating the dataset

evs = vcat(repeat([DosageRegimen(100, addl = 9, ii = 24)], 12),repeat([DosageRegimen(200, addl = 9, ii = 24)], 12))
cvs = ((WT = round(clamp(rand(Normal(80, 30)), 50, 110), digits = 2),
        SEX = rand(Binomial(1, 0.7)),
        BCRCL = round(clamp(rand(Normal(90, 20)), 40, 120), digits = 2))
        for subj in 1:24)
population = Population(map((id, cvs, evs) -> Subject(id = id, cvs = cvs, evs = evs),1:24,cvs,evs))


model = @model begin

    @param begin
        θ ∈ VectorDomain(12, lower=zeros(12), init=ones(12))
        Ω ∈ PSDDomain(12)
        Σ_dv ∈ RealDomain(lower=0.0, init=1.0)
        Σ_pddv ∈ RealDomain(lower=0.0, init=1.0)
    end

    @random begin
      η ~ MvNormal(Matrix{Float64}(I, 9, 9))
    end

    @pre begin
        Ka      = θ[1]*exp(η[1])
        CL      = θ[2]*((WT/70)^0.75)*((BCRCL/93)^0.5716)*exp(η[2])
        Vc      = θ[3]*(WT/70)^1.0*exp(η[3])
        Q       = θ[4]*(WT/70)^0.75*exp(η[4])
        Vp      = θ[5]*(WT/70)^1.0*exp(η[5])
        bioava  = θ[6]*exp(η[6])

        Kin     = θ[7]*exp(η[7])
        Kout    = θ[8]*exp(η[8])
        IC50    = θ[9]*exp(η[9])
        IMAX    = θ[10]
        γ       = θ[11]
        #CRCLexp = θ[12]
        #Base    = Kin/Kout

    end

    @covariates WT BCRCL SEX

    @init begin
        Resp = Kin/Kout
    end

    @dynamics begin
        Gut'    = -Ka*Gut
        Cent'   =  Ka*Gut - CL*(Cent/Vc) -Q*(Cent/Vc) + Q*(Periph/Vp)
        Periph' =  Q*(Cent/Vc)  - Q*(Periph/Vp)
        Resp'   =  Kin*(1-(IMAX*(Cent/Vc)^γ/(IC50^γ+(Cent/Vc)^γ)))  - Kout*Resp
    end

    @vars begin
    cp = Cent/Vc
    resp   = Resp
    #auc = AUC(cp)
end

    @derived begin
        cmax = maximum(cp)
        dv ~ @. Normal(cp, cp*Σ_dv)
        pddv ~ @. Normal(resp, Σ_pddv)

    end
end


θ = [
              1, # Ka1  Absorption rate constant 1 (1/time)
              1, # CL   Clearance (volume/time)
              20, # Vc   Central volume (volume)
              2, # Q    Inter-compartmental clearance (volume/time)
              10, # Vp   Peripheral volume of distribution (volume)
              0.6, # F1
              10, # Kin  Response in rate constant (1/time)
              2, # Kout Response out rate constant (1/time)
              2, # IC50 Concentration for 50% of max inhibition (mass/volume)
              1, # IMAX Maximum inhibition
              2  # γ    Emax model sigmoidicity
              #0.5716 # CRCL effect on CL
              ]

param = (θ = θ,
    Ω = diagm(0 => [0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04]),
    Σ_dv = 0.04,
    Σ_pddv = 1)

@test_throws ArgumentError simobs(model, population[1], param, obstimes=Float64[])
@test_throws ArgumentError conditional_nll(model, population[1], param, (η=zeros(9),))
@test_throws MethodError conditional_nll(model, population[1])
@test_nowarn simobs(model, population[1], param,obstimes=0.1:0.1:300.0)
