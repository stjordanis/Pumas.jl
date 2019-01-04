using PuMaS, Test, Random, LinearAlgebra, LabelledArrays

# Read the data# Read the data
data = process_nmtran(example_nmtran_data("data1"),
                      [:sex,:wt,:etn])
# Cut off the `t=0` pre-dose observation as it throws conditional_nll calculations
# off the scale (variance of the simulated distribution is too small).
for subject in data.subjects
    obs1 = subject.observations[1]
    if obs1.time == 0
        popfirst!(subject.observations)
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

    @dynamics begin
        cp       =  Central/V
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

function col_f(p,rfx,cov)
    (Σ  = p.Σ,
    Ka = p.θ[1],  # pre
    CL = p.θ[2] * ((cov.wt/70)^0.75) *
         (p.θ[4]^cov.sex) * exp(rfx.η[1]),
    V  = p.θ[3] * exp(rfx.η[2]))
end

OneCompartmentVector = @SLVector (:Depot,:Central)
function init_f(col,t0)
     OneCompartmentVector(0.0,0.0)
end

function static_onecompartment_f(u,p,t)
    OneCompartmentVector(-p.Ka*u[1], p.Ka*u[1] - (p.CL/p.V)*u[2])
end
prob = ODEProblem(static_onecompartment_f,nothing,nothing,nothing)

function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol)
    conc = @. central / col.V
    ___dv = @. Normal(conc, conc*col.Σ)
    dv = @. rand(___dv)
    (dv = dv,), (dv=___dv,)
end

mstatic = PKPDModel(p,rfx_f,col_f,init_f,prob,derived_f)

x0 = init_param(mdsl)
y0 = init_random(mdsl, x0)

subject = data.subjects[1]

@test conditional_nll(mdsl,subject,x0,y0,abstol=1e-12,reltol=1e-12) ≈ conditional_nll(mstatic,subject,x0,y0,abstol=1e-12,reltol=1e-12)

@test (Random.seed!(1); simobs(mdsl,subject,x0,y0,abstol=1e-12,reltol=1e-12)[:dv]) ≈
      (Random.seed!(1); simobs(mstatic,subject,x0,y0,abstol=1e-12,reltol=1e-12)[:dv])


p = ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),
              Ω = PSDDomain(2))),

function col_f2(p,rfx,cov)
    (Ka = p.θ[1],
     CL = p.θ[2] * exp(rfx.η[1]),
     V  = p.θ[3] * exp(rfx.η[2]))
end

function post_f(col,u,t)
    (conc = u[2] / col.V,)
end
function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol)
    conc = @. central / col.V
    (conc = conc,), nothing
end

mstatic2 = PKPDModel(p,rfx_f,col_f2,init_f,prob,derived_f)

subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))

x0 = (θ = [
              1.5,  #Ka
              1.0,  #CL
              30.0 #V
              ],
         Ω = Matrix{Float64}(I, 2, 2))
y0 = (η = zeros(2),)

sol = solve(mstatic2,subject,x0,y0;obstimes=[i*12+1e-12 for i in 0:1],abstol=1e-12,reltol=1e-12)

p = simobs(mstatic2,subject,x0,y0;obstimes=[i*12+1e-12 for i in 0:1],abstol=1e-12,reltol=1e-12)
@test 1000p[:conc] ≈ [605.3220736386598;1616.4036675452326]
