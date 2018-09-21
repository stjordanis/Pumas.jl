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
        θ1 := θ[1]
        Ka = θ1
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

### Function-Based Interface

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              Σ = RealDomain(lower=0.0, init=1.0),
              a = ConstDomain(0.2)))

function rfx_f(p)
    RandomEffectSet((η=MvNormal(p.Ω),))
end

function col_f(p,rfx,cov)
    (Σ  = p.Σ,
    Ka = p.θ[1],  # collate
    CL = p.θ[2] * ((cov.wt/70)^0.75) *
         (p.θ[4]^cov.sex) * exp(rfx.η[1]),
    V  = p.θ[3] * exp(rfx.η[2]))
end

function init_f(col,t0)
    [0.0,0.0]
end

function onecompartment_f(du,u,p,t)
    du[1] = -p.Ka*u[1]
    du[2] =  p.Ka*u[1] - (p.CL/p.V)*u[2]
end

function post_f(col,u,t)
    conc = u[2] / col.V
    (dv = Normal(conc, conc*col.Σ),)
end

mobj = PKPDModel(p,rfx_f,col_f,init_f,onecompartment_f,post_f)

x0 = init_param(mdsl)
y0 = init_random(mdsl, x0)

subject = data.subjects[1]

sol1 = solve(mdsl,subject,x0,y0)
sol2 = solve(mobj,subject,x0,y0)

@test likelihood(mdsl,subject,x0,y0) ≈ likelihood(mobj,subject,x0,y0)

@test (Random.seed!(1); map(x -> x.dv, simobs(mdsl,subject,x0,y0))) ≈
      (Random.seed!(1); map(x -> x.dv, simobs(mobj,subject,x0,y0)))
