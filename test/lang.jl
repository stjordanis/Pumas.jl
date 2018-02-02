using Base.Test
using PKPDSimulator, NamedTuples, PDMats, Distributions

# Read the data
covariates = [:sex,:wt,:etn]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,separator=',')

# add a small epsilon to time 0 observations
for subject in data.subjects
    obs1 = subject.observations[1]
    if obs1.time == 0
        subject.observations[1] = PKPDSimulator.Observation(sqrt(eps()), obs1.val, obs1.cmt)
    end
end


## parameters
params = ParamSet(@NT(θ=VectorDomain(4, lower=zeros(4),init=ones(4)), Ω=PSDDomain(2), Σ=RealDomain(lower=0.0,init=1.0), a=ConstDomain(0.2)))
x0 = init(params)

n = PKPDSimulator.packlen(params)
v = zeros(n)

PKPDSimulator.pack!(v, params, x0)
v .+= 0.1
x1 = PKPDSimulator.unpack(v, params)

## random effects
function randomfx(p)
    RandomEffectSet(@NT(η=RandomEffect(MvNormal(p.Ω))))
end
rfx = randomfx(x0)
y0 = init(rfx)
m = PKPDSimulator.packlen(rfx)
w = zeros(m)
PKPDSimulator.pack!(w, rfx, y0)
w .+= 0.1
y1 = PKPDSimulator.unpack(w, rfx)

## parameter reduction
# this is calculated once per subject
function collate(params, randoms, covars)
    θ = params.θ
    η = randoms.η
    wt = covars.wt
    sex = covars.sex
    @NT(Ka = θ[1],
        CL = θ[2]*((wt/70)^0.75)*(θ[4]^sex)*exp(η[1]),
        V  = θ[3]*exp(η[2]))
end


## ODE
# computed at each iteration of the ODE solver
function f(du,u,p,t)
    Depot, Central = u    
    du[1] = dDepot   = -p.Ka*Depot
    du[2] = dCentral =  p.Ka*Depot - (p.CL/p.V)*Central
end
u0 = zeros(2)

odeprob = raw_pkpd_problem(f, u0)

## distributions
# computed at each observation
function err(params, randoms, covars, u,p, t)
    V = p.V
    Depot, Central = u    
    Σ = params.Σ
    conc = Central / V
    @NT(dv=Normal(conc, conc*Σ))
end


model = NLModel(params,randomfx,collate,odeprob,err)
subject = data.subjects[1]

pkpd_simulate(model,subject,x0,y0)
pkpd_likelihood(model,subject,x0,y0)
