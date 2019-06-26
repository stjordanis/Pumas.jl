using PuMaS
using LinearAlgebra

choose_covariates() = (isPM = rand([1, 0]),
                       Wt = rand(55:80))

function generate_population(events,nsubs=4)
  pop = Population(map(i -> Subject(id=i,evs=events,cvs=choose_covariates()),1:nsubs))
  return pop
end

ev = DosageRegimen(100, cmt = 2)
ev2 = generate_population(ev)

m_diffeq = @model begin
  @param   begin
    θ ∈ VectorDomain(5, lower=zeros(5), init=ones(5))
    Ω ∈ PSDDomain(2)
    σ_prop ∈ RealDomain(init=0.1)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    Ka = θ[1]
    CL = θ[2]*exp(η[1])
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
    nca := @nca cp
    auc =  NCA.auc(nca)
    thalf =  NCA.thalf(nca)
    cmax = NCA.cmax(nca)
  end
end

p = (  θ = [1.5,  #Ka
            1.1,  #CL
            20.0,  #V
            0, # lags2
            1 #Bioav
           ],
     Ω = diagm(0 => [0.04,0.04]),
     σ_prop = 0.00
    )


sim = @test_nowarn simobs(m_diffeq, ev2, p; abstol=1e-14, reltol=1e-14, parallel_type=PuMaS.Serial)
for i in eachindex(sim.sims)
  @test NCA.auc(sim[i].observed.cp, sim[i].times) === sim[i].observed.auc
  @test NCA.thalf(sim[i].observed.cp, sim[i].times) === sim[i].observed.thalf
  @test NCA.cmax(sim[i].observed.cp, sim[i].times) === sim[i].observed.cmax
end

pop = Population(map(i->sim.sims[i].subject, eachindex(sim.sims)))
@test_nowarn NCAPopulation(pop, name=:cp, verbose=false)
@test_nowarn NCASubject(pop[1], name=:cp)
@test NCADose(ev2[1].events[1]) === NCADose(0.0, 100.0, 0.0, NCA.IVBolus)

ev = DosageRegimen(2000, ii=24, addl=3)
ev1 = generate_population(ev)
parmet = @model begin
    @param begin
        θ = VectorDomain(5)
        Ω = VectorDomain(4)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ[1]*exp(η[1])
        V = θ[2]*exp(η[2])
        CLM = θ[3]*exp(η[3])
        VM = θ[4]*exp(η[4])
        fm = θ[5]
        K = (fm*(CL/V))
        KPM =  ((1-fm)*(CL/V))
        KM = (CLM/VM)
    end

    @dynamics begin
        CENT'   = -K*CENT - KPM*CENT
        METAB'  =  KPM*CENT - KM*METAB
    end

    @derived begin
        cp = @. CENT/V
        cm = @. METAB/VM

        ncas := @nca cp cm

        auccp, auccm = @. NCA.auc(ncas)
        thalfcp, thalfcm = @. NCA.thalf(ncas)
        cmaxcp, cmaxcm = @. NCA.cmax(ncas)
    end
end

p = (  θ = [11.5,  #CL
           50.0,  #V
           10, #CLM
           8, #VM
           0.7
           ],
      Ω = diagm(0 => [0.04,0.04,0.04,0.04])
      )
sim = @test_nowarn simobs(parmet, ev1, p)
@test_nowarn DataFrame(sim)
dose = NCADose.(sim[1].subject.events)
for i in eachindex(sim.sims)
  subjcp = NCASubject(sim[i].observed.cp, sim[i].times, dose=dose, clean=false)
  subjcm = NCASubject(sim[i].observed.cm, sim[i].times, dose=dose, clean=false)
  @test NCA.auc(subjcp) == sim[i].observed.auccp
  @test NCA.thalf(subjcp) == sim[i].observed.thalfcp
  @test NCA.cmax(subjcp) == sim[i].observed.cmaxcp
  @test NCA.auc(subjcm) == sim[i].observed.auccm
  @test NCA.thalf(subjcm) == sim[i].observed.thalfcm
  @test NCA.cmax(subjcm) == sim[i].observed.cmaxcm
end

pop = Population(map(i->sim.sims[i].subject, eachindex(sim.sims)))
@test_nowarn NCAPopulation(pop, name=:cp, verbose=false)
@test_nowarn NCASubject(pop[1], name=:cp)
@test NCADose(ev1[1].events[1]) === NCADose(0.0, 2000.0, 0.0, NCA.IVBolus)
