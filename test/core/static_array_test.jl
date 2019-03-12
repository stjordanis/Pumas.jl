using PuMaS, Test, Random, LinearAlgebra, LabelledArrays

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
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn

    @pre begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @vars begin
        cp = Central/V
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @derived begin
        conc = @. Central / V
        dv ~ @. Normal(conc, conc*Σ)
    end
end

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              Σ = RealDomain(lower=0.0, init=1.0),
              a = ConstDomain(0.2)))

function rfx_f(p)
    ParamSet((η=MvNormal(p.Ω),))
end

function col_f(fixeffs,randeffs,subject)
    cov = subject.covariates
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

function static_onecompartment_f(u,p,t)
    OneCompartmentVector(-p.Ka*u[1], p.Ka*u[1] - (p.CL/p.V)*u[2])
end
prob = ODEProblem(static_onecompartment_f,nothing,nothing,nothing)

function derived_f(col,sol,obstimes,subject)
    central = sol(obstimes;idxs=2)
    conc = @. central / col.V
    dv = @. Normal(conc, conc*col.Σ)
    (dv = dv,)
end

observed_f(col,sol,obstimes,samples,subject) = samples

mstatic = PuMaSModel(p,rfx_f,col_f,init_f,prob,derived_f,observed_f)

fixeffs = init_fixeffs(mdsl)
randeffs = init_randeffs(mdsl, fixeffs)

subject = data.subjects[1]

@test conditional_nll(mdsl,subject,fixeffs,randeffs,abstol=1e-12,reltol=1e-12) ≈ conditional_nll(mstatic,subject,fixeffs,randeffs,abstol=1e-12,reltol=1e-12)

@test (Random.seed!(1); simobs(mdsl,subject,fixeffs,randeffs,abstol=1e-12,reltol=1e-12)[:dv]) ≈
      (Random.seed!(1); simobs(mstatic,subject,fixeffs,randeffs,abstol=1e-12,reltol=1e-12)[:dv])


p = ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),
              Ω = PSDDomain(2))),

function col_f2(fixeffs,randeffs,subject)
    (Ka = fixeffs.θ[1],
     CL = fixeffs.θ[2] * exp(randeffs.η[1]),
     V  = fixeffs.θ[3] * exp(randeffs.η[2]))
end

function post_f(col,u,t)
    (conc = u[2] / col.V,)
end
function derived_f(col,sol,obstimes,subject)
    central = sol(obstimes;idxs=2)
    conc = @. central / col.V
    (conc = conc,)
end

mstatic2 = PuMaSModel(p,rfx_f,col_f2,init_f,prob,derived_f,observed_f)

subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))

fixeffs = (θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0 #V
              ],
         Ω = Matrix{Float64}(I, 2, 2))
randeffs = (η = zeros(2),)

sol = solve(mstatic2,subject,fixeffs,randeffs;obstimes=[i*12+1e-12 for i in 0:1],abstol=1e-12,reltol=1e-12)

p = simobs(mstatic2,subject,fixeffs,randeffs;obstimes=[i*12+1e-12 for i in 0:1],abstol=1e-12,reltol=1e-12)
@test 1000p[:conc] ≈ [605.3220736386598;1616.4036675452326]
