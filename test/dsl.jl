using Test

using PuMaS, Distributions, ParameterizedFunctions, Random


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
        a ∈ ConstDomain(0.2)
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
        conc := Central / V
        dv ~ Normal(conc, conc*Σ)
    end
end

@isdefined(mobj) || (mobj = PKPDModel(ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
                              Ω = PSDDomain(2),
                              Σ = RealDomain(lower=0.0, init=1.0),
                              a = ConstDomain(0.2))),
                 (_param) -> RandomEffectSet((η = MvNormal(_param.Ω),)), # random effects
                 (_param, _random, _covariates) -> (Σ  = _param.Σ,
                                                     Ka = _param.θ[1],  # collate
                                                     CL = _param.θ[2] * ((_covariates.wt/70)^0.75) *
                                                          (_param.θ[4]^_covariates.sex) * exp(_random.η[1]),
                                                     V  = _param.θ[3] * exp(_random.η[2])),
                 (_pre,t0) -> [0.0,0.0], # init
                 ParameterizedFunctions.@ode_def(OneCompartment,begin # dynamics
                     dDepot   = -Ka*Depot
                     dCentral =  Ka*Depot - (CL/V)*Central
                 end, Σ, Ka, CL, V),
                 (_pre,_odevars,t) -> (conc = _odevars[2] / _pre.V; # error
                                       (dv = Normal(conc, conc*_pre.Σ),))))

x0 = init_param(mdsl)
y0 = init_random(mdsl, x0)

subject = data.subjects[1]

@test likelihood(mdsl,subject,x0,y0) ≈ likelihood(mobj,subject,x0,y0)

@test (Random.seed!(1); map(x -> x.dv, simobs(mdsl,subject,x0,y0))) ≈
      (Random.seed!(1); map(x -> x.dv, simobs(mobj,subject,x0,y0)))
