using Test

using PuMaS, Distributions, ParameterizedFunctions, Random, LabelledArrays


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

    @vars begin
      conc = Central / V
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*conc
    end

    @post begin
        dv ~ Normal(conc, conc*Σ)
    end

    @derived begin
      obs_cmax = maximum(dv)
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

@SLVector OneCompartmentVector Float64 [Depot,Central]

function init_f(col,t0)
    OneCompartmentVector(0.0,0.0)
end

function onecompartment_f(u,p,t)
    OneCompartmentVector(-p.Ka*u[1],
                          p.Ka*u[1] - (p.CL/p.V)*u[2])
end

function post_f(col,u,t)
    conc = u[2] / col.V
    (dv = Normal(conc, conc*col.Σ),)
end

function derived_f(col,sol,obstimes,obs)
    (obs_cmax = maximum(obs[:dv]),)
end

mobj = PKPDModel(p,rfx_f,col_f,init_f,onecompartment_f,post_f,derived_f)

x0 = init_param(mdsl)
y0 = init_random(mdsl, x0)

subject = data.subjects[1]

sol1 = solve(mdsl,subject,x0,y0)
sol2 = solve(mobj,subject,x0,y0)

@test sol1[10].Central ≈ sol2[10].Central

@test likelihood(mdsl,subject,x0,y0) ≈ likelihood(mobj,subject,x0,y0)

@test simobs(mdsl,subject,x0,y0).derived.obs_cmax == simobs(mobj,subject,x0,y0).derived.obs_cmax

@test (Random.seed!(1); simobs(mdsl,subject,x0,y0)[:dv]) ≈
      (Random.seed!(1); simobs(mobj,subject,x0,y0)[:dv])

# Now test an array-based version

function init_f_iip(col,t0)
    [0.0,0.0]
end

function onecompartment_f_iip(du,u,p,t)
    du[1] = -p.Ka*u[1]
    du[2] =  p.Ka*u[1] - (p.CL/p.V)*u[2]
end

mobj_iip = PKPDModel(p,rfx_f,col_f,init_f_iip,onecompartment_f_iip,post_f,derived_f)
sol2 = solve(mobj_iip,subject,x0,y0)

@test likelihood(mobj_iip,subject,x0,y0) ≈ likelihood(mobj,subject,x0,y0) rtol=1e-4

@test (Random.seed!(1); simobs(mobj_iip,subject,x0,y0)[:dv]) ≈
      (Random.seed!(1); simobs(mobj,subject,x0,y0)[:dv]) rtol=1e-4
