using PuMaS, Test, Random, LabelledArrays


# Read the data# Read the data
data = process_nmtran(example_nmtran_data("data1"),
                      [:sex,:wt,:etn])
# Cut off the `t=0` pre-dose observation as it throws conditional_nll calculations
# off the scale (variance of the simulated distribution is too small).
for subject in data.subjects
    if subject.time[1] == 0
        popfirst!(subject.time)
        popfirst!(subject.observations.dv)
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

    @pre begin
        θ1 := θ[1]
        Ka = θ1
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @vars begin
      conc = Central / V
    end

    @dynamics begin
        Depot'   := -Ka*Depot # test for `:=` handling
        Central' =  Ka*Depot - CL*conc
    end

    @derived begin
      dv ~ @. Normal(conc, conc*Σ)
      T_max = maximum(t)
    end

    @observed begin
      obs_cmax = maximum(dv)
    end
end

### Function-Based Interface

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              Σ = RealDomain(lower=0.0, init=1.0),
              a = ConstDomain(0.2)))

function rfx_f(p)
    ParamSet((η=MvNormal(p.Ω),))
end

function col_f(fixeffs,randeffs,cov)
    (Σ  = fixeffs.Σ,
    Ka = fixeffs.θ[1],  # pre
    CL = fixeffs.θ[2] * ((cov.wt/70)^0.75) *
         (fixeffs.θ[4]^cov.sex) * exp(randeffs.η[1]),
    V  = fixeffs.θ[3] * exp(randeffs.η[2]))
end

OneCompartmentVector = @SLVector (:Depot,:Central)

function init_f(col,t0)
    OneCompartmentVector(0.0,0.0)
end

function onecompartment_f(u,p,t)
    OneCompartmentVector(-p.Ka*u[1],
                          p.Ka*u[1] - (p.CL/p.V)*u[2])
end
prob = ODEProblem(onecompartment_f,nothing,nothing,nothing)

# In the function interface, the first return value is a named tuple of sampled
# values, the second is a named tuple of distributions
function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol)
    conc = @. central / col.V
    dv = @. Normal(conc, conc*col.Σ)
    (dv=dv,)
end

function observed_f(col,sol,obstimes,samples)
    (obs_cmax = maximum(samples.dv),
     T_max = maximum(obstimes),
     dv = samples.dv)
end

mobj = PuMaSModel(p,rfx_f,col_f,init_f,prob,derived_f,observed_f)

fixeffs = init_param(mdsl)
randeffs = init_random(mdsl, fixeffs)

subject = data.subjects[1]

sol1 = solve(mdsl,subject,fixeffs,randeffs)
sol2 = solve(mobj,subject,fixeffs,randeffs)

@test sol1[10].Central ≈ sol2[10].Central

@test conditional_nll(mdsl,subject,fixeffs,randeffs) ≈ conditional_nll(mobj,subject,fixeffs,randeffs) rtol=5e-3

Random.seed!(1); obs_dsl = simobs(mdsl,subject,fixeffs,randeffs)
Random.seed!(1); obs_obj = simobs(mobj,subject,fixeffs,randeffs)

@test obs_dsl.observed.obs_cmax == obs_obj.observed.obs_cmax > 0
@test obs_dsl.observed.T_max == obs_obj.observed.T_max

@test obs_dsl[:dv] ≈ obs_obj[:dv]

# Now test an array-based version

function init_f_iip(col,t0)
    [0.0,0.0]
end

function onecompartment_f_iip(du,u,p,t)
    du[1] = -p.Ka*u[1]
    du[2] =  p.Ka*u[1] - (p.CL/p.V)*u[2]
end
prob = ODEProblem(onecompartment_f_iip,nothing,nothing,nothing)

mobj_iip = PuMaSModel(p,rfx_f,col_f,init_f_iip,prob,derived_f,observed_f)
sol2 = solve(mobj_iip,subject,fixeffs,randeffs)

@test conditional_nll(mobj_iip,subject,fixeffs,randeffs) ≈ conditional_nll(mobj,subject,fixeffs,randeffs) rtol=5e-3

@test (Random.seed!(1); simobs(mobj_iip,subject,fixeffs,randeffs)[:dv]) ≈
      (Random.seed!(1); simobs(mobj,subject,fixeffs,randeffs)[:dv]) rtol=1e-4

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

    @derived begin
      dv ~ [Binomial(30,Ka*CL) for i in 1:length(t)]
    end

    @observed begin
        obs_cmax = maximum(dv)
        T_max = maximum(t)
    end
end
fixeffs = init_param(mdsl)
randeffs = init_random(mdsl, fixeffs)

@test solve(mdsl,subject,fixeffs,randeffs) === nothing
@test simobs(mdsl,subject,fixeffs,randeffs) != nothing
@test conditional_nll(mdsl,subject,fixeffs,randeffs) == Inf # since real-valued observations
