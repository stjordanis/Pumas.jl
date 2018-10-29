using PuMaS, ForwardDiff, DiffEqDiffTools, Test
AD_gradient = ForwardDiff.gradient
AD_hessian = ForwardDiff.hessian
FD_gradient = DiffEqDiffTools.finite_difference_gradient
FD_jacobian = DiffEqDiffTools.finite_difference_jacobian
FD_hessian = function (f, x)
    grad_fun = y -> FD_gradient(f, y)
    FD_jacobian(grad_fun, x, Val{:central}, eltype(x), Val{false})
end

subject = process_nmtran(example_nmtran_data("event_data/data2"),
                         [], [:cp])[1]

# Simple one-compartment model (uses static vector)
model = @model begin
    @param   θ ∈ VectorDomain(3, lower=zeros(3), init=ones(3))
    @random  η ~ MvNormal(Matrix{Float64}(I, 2, 2))

    @pre begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/V)*Central
    end

    @derived begin
        cp = @. Central / V
        cmax = maximum(cp)
        pp = exp(cmax) / (1 + exp(cmax)) # pain probability
    end
end

# Function-Based Interface (non-static vector)
p = ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),))
rfx_f(p) = RandomEffectSet((η=MvNormal(Matrix{Float64}(I, 2, 2)),))
function col_f(p,rfx,cov)
    (Ka = p.θ[1],
    CL = p.θ[2] * exp(rfx.η[1]),
    V  = p.θ[3] * exp(rfx.η[2]))
end
init_f(col,t0) = [0.0,0.0]
function onecompartment_f(du,u,p,t)
    du[1] = -p.Ka*u[1]
    du[2] = p.Ka*u[1] - (p.CL/p.V)*u[2]
end
prob = ODEProblem(onecompartment_f,nothing,nothing,nothing)
function derived_f(col,sol,obstimes)
    central = map(x->x[2], sol)
    cp = @. central / col.V
    cmax = maximum(cp)
    pp = exp(cmax) / (1 + exp(cmax))
    (cmax = cmax, pp = pp), ()
end
model_ip = PKPDModel(p,rfx_f,col_f,init_f,prob,derived_f)

# Test gradient and hessian of an observable w.r.t. θ
function test_fun(model, observable)
    function (θ)
        x0 = (θ = θ,)
        y0 = (η = eltype(θ).([0.0,0.0]),)
        sim = simobs(model, subject, x0, y0; abstol=1e-14, reltol=1e-14)
        sim[observable]
    end
end

θ0 = [1.5, 1.0, 30.0]
# Static vectors currently breaks due to type conversion error
fun = test_fun(model, :cmax)
@test_broken FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-10
@test_broken FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-4
fun = test_fun(model, :pp)
@test_broken FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-10
@test_broken FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-4

fun = test_fun(model_ip, :cmax)
@test FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-10
@test FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-4
fun = test_fun(model_ip, :pp)
@test FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-10
@test FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-4
