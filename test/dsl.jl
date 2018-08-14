using Test

using PuMaS, Distributions, NamedTuples


# Read the data# Read the data
data = process_data(joinpath(Pkg.dir("PuMaS"),"examples/data1.csv"),
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

    @data_cov sex wt etn

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
    end

    @error begin
        conc = Central / V
        dv ~ Normal(conc, conc*Σ)
    end
end

mobj = PKPDModel(ParamSet(@NT(θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
                              Ω = PSDDomain(2),
                              Σ = RealDomain(lower=0.0, init=1.0),
                              a = ConstDomain(0.2))),
                 (_param) -> RandomEffectSet(@NT(η = MvNormal(_param.Ω))), # random effects
                 (_param, _random, _data_cov) -> @NT(Σ  = _param.Σ,
                                                     Ka = _param.θ[1],  # collate
                                                     CL = _param.θ[2] * ((_data_cov.wt/70)^0.75) *
                                                          (_param.θ[4]^_data_cov.sex) * exp(_random.η[1]),
                                                     V  = _param.θ[3] * exp(_random.η[2])),
                 (_pre,t0) -> [0.0,0.0], # init
                 ParameterizedFunctions.@ode_def(OneCompartment,begin # dynamics
                     dDepot   = -Ka*Depot
                     dCentral =  Ka*Depot - (CL/V)*Central
                 end, Σ, Ka, CL, V),
                 (_pre,_odevars,t) -> @NT(conc = _odevars[2] / _pre.V), # post
                 (_pre,_odevars,t) -> (conc = _odevars[2] / _pre.V; # error
                                                     @NT(dv = Normal(conc, conc*_pre.Σ))))
                 
                 

x0 = init_param(mdsl)
y0 = init_random(mdsl, x0)

subject = data.subjects[1]

@test pkpd_likelihood(mdsl,subject,x0,y0) ≈ pkpd_likelihood(mobj,subject,x0,y0)

@test (srand(1); map(x -> x.dv, pkpd_simulate(mdsl,subject,x0,y0))) ≈
      (srand(1); map(x -> x.dv, pkpd_simulate(mobj,subject,x0,y0)))

@test map(x -> x.conc, pkpd_post(mdsl,subject,x0,y0)) ≈
      map(x -> x.conc, pkpd_post(mobj,subject,x0,y0))

post_dsl = pkpd_postfun(mdsl, subject, x0, y0)
post_obj = pkpd_postfun(mobj, subject, x0, y0)

@test post_dsl(1).conc ≈ post_obj(1).conc
