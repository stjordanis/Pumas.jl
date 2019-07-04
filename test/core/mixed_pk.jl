using Pumas, Test, Random, LabelledArrays, StaticArrays

# Read the data# Read the data
data = read_pumas(example_nmtran_data("data1"),
                      cvs = [:sex,:wt,:etn])
# Cut off the `t=0` pre-dose observation as it throws conditional_nll calculations
# off the scale (variance of the simulated distribution is too small).
for subject in data
    if subject.time[1] == 0
        popfirst!(subject.time)
        popfirst!(subject.observations.dv)
    end
end

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              Σ = RealDomain(lower=0.0, init=1.0),
              a = ConstDomain(0.2)))

function rfx_f(p)
    ParamSet((η=MvNormal(p.Ω),))
end

function col_f(param,randeffs,subject)
    (Σ  = param.Σ,
    Ka = param.θ[1],  # pre
    CL = param.θ[2] * ((subject.covariates.wt/70)^0.75) *
         (param.θ[4]^subject.covariates.sex) * exp(randeffs.η[1]),
    V  = param.θ[3] * exp(randeffs.η[2]))
end

function init_f(col,t0)
    @SVector [0.0]
end

function onecompartment_f(u,p,t)
    Depot, Central = p.___pk(t)
    @SVector [Depot-Central]
end

pkprob = OneCompartmentModel()
prob2 = ODEProblem(onecompartment_f,nothing,nothing,nothing)
prob = AnalyticalPKProblem(pkprob,prob2)

# In the function interface, the first return value is a named tuple of sampled
# values, the second is a named tuple of distributions
function derived_f(col,sol,obstimes,subject)
    central2 = sol(obstimes;idxs=1)
    conc = @. central / col.V
    dv = @. Normal(conc, conc*col.Σ)
    (dv=dv,)
end

function observed_f(col,sol,obstimes,samples,subject)
    (obs_cmax = maximum(samples.dv),
     T_max = maximum(obstimes),
     dv = samples.dv)
end

mobj = PumasModel(p,rfx_f,col_f,init_f,prob,derived_f,observed_f)

param = (θ = ones(4), Ω = [1.0 0.0
                           0.0 1.0], Σ = 1.0, a = 0.2)
randeffs = (η = zeros(2),)
subject = data[1]
sol1 = solve(mobj,subject,param,randeffs,abstol=1e-12,reltol=1e-12)

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

    @pre begin
        θ1 := θ[1]
        Ka = θ1
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/V)*Central
        Res' = Depot - Central
    end

    @derived begin
      dv ~ @. Normal(conc, conc*Σ)
      T_max = maximum(t)
    end

    @observed begin
      obs_cmax = maximum(dv)
    end
end
sol2 = solve(mdsl,subject,param,randeffs,abstol=1e-12,reltol=1e-12)

t = 0.25:0.25:19.0
@test all(sol1(t[i])[3] .- sol2(t[i])[3] < 1e-8 for i in 1:length(t))
@test all(sol1.numsol[i][1] .- sol2[i][3] < 1e-8 for i in 1:length(sol1.numsol))
@test all(sol1(t) .- sol2(t) .< 1e-8)
