using PuMaS, ForwardDiff, DiffEqDiffTools, Test, Random

AD_gradient = ForwardDiff.gradient
AD_hessian = ForwardDiff.hessian
FD_gradient = DiffEqDiffTools.finite_difference_gradient
FD_jacobian = DiffEqDiffTools.finite_difference_jacobian
FD_hessian = function (f, x)
    grad_fun = y -> FD_gradient(f, y)
    FD_jacobian(grad_fun, x, Val{:central}, eltype(x), Val{false})
end

data = process_nmtran(example_nmtran_data("data1"),
                      [:sex,:wt,:etn])
subject = data.subjects[1]
# Cut off the `t=0` pre-dose observation as it throws conditional_loglikelihood calculations
# off the scale (variance of the simulated distribution is too small).
for subject in data.subjects
    obs1 = subject.observations[1]
    if obs1.time == 0
        popfirst!(subject.observations)
    end
end

# Simple one-compartment model (uses static vector)
model = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        σ ∈ RealDomain(lower=0.0, init=1.0)
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
        cmax = maximum(conc)
        dv ~ @. Normal(conc, conc*σ)
    end
end

# Function-Based Interface (non-static vector)
p = ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),
              Ω = PSDDomain(2),
              σ = RealDomain(lower=0.0, init=1.0)))
rfx_f(p) = RandomEffectSet((η=MvNormal(p.Ω),))
function col_f(p,rfx,cov)
    (Ka = p.θ[1],
     CL = p.θ[2] * ((cov.wt/70)^0.75) * (p.θ[4]^cov.sex) * exp(rfx.η[1]),
     V  = p.θ[3] * exp(rfx.η[2]),
     σ = p.σ)
end
init_f(col,t0) = [0.0,0.0]
function onecompartment_f(du,u,p,t)
    Depot, Central = u # TODO: use field access after LabelledArray fix
    cp = Central/p.V
    du[1] = -p.Ka*Depot
    du[2] = p.Ka*Depot - p.CL*cp
end
prob = ODEProblem(onecompartment_f,nothing,nothing,nothing)
function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol) # TODO: use field access after LabelledArray fix
    _conc = @. central / col.V
    _cmax = maximum(_conc)
    _dv = @. Normal(_conc, _conc*col.σ)
    (conc = _conc, cmax = _cmax), (dv = _dv,)
end
model_ip = PKPDModel(p,rfx_f,col_f,init_f,prob,derived_f)

# Initial data
θ0 = [2.268,74.17,468.6,0.5876]
x0 = (θ = θ0, Ω = PDMat([0.05 0.0; 0.0 0.2]), σ = 0.1)
Random.seed!(0); y0 = init_random(model, x0)

# Test gradient and hessian of an observable w.r.t. θ
function test_obs(model)
    function (θ)
        _x0 = (θ = θ, Ω = x0.Ω, σ = x0.σ)
        sim = simobs(model, subject, _x0, y0; abstol=1e-14, reltol=1e-14)
        sim[:cmax]
    end
end

# Static vectors currently breaks due to type conversion error
fun = test_obs(model)
@test_broken FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-8
@test_broken FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-3

fun = test_obs(model_ip)
@test FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-8
@test FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-3

# Test gradient and hessian of the total log conditional_loglikelihood w.r.t. θ
function test_conditional_loglikelihood(model)
    function (θ)
        _x0 = (θ = θ, Ω = x0.Ω, σ = x0.σ)
        conditional_loglikelihood(model, subject, _x0, y0; abstol=1e-14, reltol=1e-14)
    end
end

fun = test_conditional_loglikelihood(model)
@test_broken FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-8
@test_broken FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-3

fun = test_conditional_loglikelihood(model_ip)
@test FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=2e-7
@test FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=5e-3

dists, val, grad, hes = PuMaS.conditional_loglikelihood_derivatives(model_ip,subject,x0,y0,:θ)

PuMaS.marginal_loglikelihood(model_ip,subject,x0,y0)
