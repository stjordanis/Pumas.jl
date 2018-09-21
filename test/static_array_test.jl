using Test

using PuMaS, Distributions, StaticArrays, Random


# Read the data# Read the data
data = process_data(joinpath(joinpath(dirname(pathof(PuMaS)), ".."),"examples/data1.csv"),
                    [:sex,:wt,:etn],separator=',')
# add a small epsilon to time 0 observations
for subject in data.subjects
    obs1 = subject.observations[1]
    if obs1.time == 0
        subject.observations[1] = PuMaS.Observation(sqrt(eps()), obs1.val, obs1.cmt)
    end
end

## parameters
mdsl = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        Σ ∈ RealDomain(lower=0.0, init=1.0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn

    @collate begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post begin
        conc = Central / V
        dv ~ Normal(conc, conc*Σ)
    end
end

mstatic = PKPDModel(ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)),
                              Ω = PSDDomain(2),
                              Σ = RealDomain(lower=0.0, init=1.0))),
                 (_param) -> RandomEffectSet((η = MvNormal(_param.Ω),)),
                 (_param, _random, _covariates) -> (Σ = _param.Σ,
                                                     Ka = _param.θ[1],
                                                     CL = _param.θ[2] * ((_covariates.wt/70)^0.75) *
                                                          (_param.θ[4]^_covariates.sex) * exp(_random.η[1]),
                                                     V  = _param.θ[3] * exp(_random.η[2])),
                 (_pre,t) -> @SVector([0.0,0.0]),
                 function depot_model(u,p,t)
                     Depot,Central = u
                     @SVector [-p.Ka*Depot,
                                p.Ka*Depot - (p.CL/p.V)*Central
                              ]
                 end,
                 (_pre,_odevars,t) -> (conc = _odevars[2] / _pre.V;
                                                     (dv = Normal(conc, conc*_pre.Σ),)))



x0 = init_param(mdsl)
y0 = init_random(mdsl, x0)

subject = data.subjects[1]

@test likelihood(mdsl,subject,x0,y0,abstol=1e-12,reltol=1e-12) ≈ likelihood(mstatic,subject,x0,y0,abstol=1e-12,reltol=1e-12)

@test (Random.seed!(1); map(x -> x.dv, simobs(mdsl,subject,x0,y0,abstol=1e-12,reltol=1e-12))) ≈
      (Random.seed!(1); map(x -> x.dv, simobs(mstatic,subject,x0,y0,abstol=1e-12,reltol=1e-12)))


#
mstatic2 = PKPDModel(ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),
                              Ω = PSDDomain(2))),
                 (_param) -> RandomEffectSet((η = MvNormal(_param.Ω),)),
                 (_param, _random, _covariates) -> (Ka = _param.θ[1],
                                                     CL = _param.θ[2] * exp(_random.η[1]),
                                                     V  = _param.θ[3] * exp(_random.η[2])),
                 (_pre,t) -> @SVector([0.0,0.0]),
                 function depot_model(u,p,t)
                     Depot,Central = u
                     @SVector [-p.Ka*Depot,
                                p.Ka*Depot - (p.CL/p.V)*Central
                              ]
                 end,
                 (_pre,_odevars,t) -> (conc = _odevars[2] / _pre.V,))



subject = build_dataset(amt=[10,20], ii=[24,24], addl=[2,2], ss=[1,2], time=[0,12],  cmt=[2,2])

x0 = (θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0 #V
              ],
         Ω = Matrix{Float64}(I, 2, 2))
y0 = (η = zeros(2),)

p = simobs(mstatic2,subject,x0,y0;obstimes=[i*12+1e-12 for i in 0:1],abstol=1e-12,reltol=1e-12)
@test [1000*x.conc for x in p] ≈ [605.3220736386598;1616.4036675452326]
