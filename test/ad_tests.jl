using PuMaS, ForwardDiff, DiffEqDiffTools, Test
AD = ForwardDiff.gradient
FD = DiffEqDiffTools.finite_difference_gradient

subject = process_nmtran(example_nmtran_data("event_data/data2"),
                         [], [:cp])[1]

# # Simple one-compartment model (uses static vector)
# # Fails now due to convertion problem with @SLVector
# model = @model begin
#     @param   θ ∈ VectorDomain(3, lower=zeros(3), init=ones(3))
#     @random  η ~ MvNormal(Matrix{Float64}(I, 2, 2))

#     @pre begin
#         Ka = θ[1]
#         CL = θ[2]*exp(η[1])
#         V  = θ[3]*exp(η[2])
#     end

#     @dynamics begin
#         Depot'   = -Ka*Depot
#         Central' =  Ka*Depot - (CL/V)*Central
#     end

#     @derived begin
#         cp = @. Central / V
#         cmax = maximum(cp)
#         pp = exp(cmax) / (1 + exp(cmax)) # pain probability
#     end
# end

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
model = PKPDModel(p,rfx_f,col_f,init_f,prob,derived_f)

# Compute the jacobian of (cmax, pp) w.r.t. θ
function test_fun(which)
    function (θ)
        x0 = (θ = θ,)
        y0 = (η = eltype(θ).([0.0,0.0]),)
        sim = simobs(model, subject, x0, y0; abstol=1e-14, reltol=1e-14)
        sim[which]
    end
end

θ0 = [1.5, 1.0, 30.0]
gcmax_FD = FD(test_fun(:cmax), θ0)
gcmax_AD = AD(test_fun(:cmax), θ0)
@test gcmax_FD ≈ gcmax_AD atol=1e-10
gpp_FD = FD(test_fun(:pp), θ0)
gpp_AD = AD(test_fun(:pp), θ0)
@test gpp_FD ≈ gpp_AD atol=1e-10
