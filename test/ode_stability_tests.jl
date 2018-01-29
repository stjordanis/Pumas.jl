using PKPDSimulator, NamedTuples, Base.Test

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), covariates,dvs)

# Define the ODE
function depot_model(du,u,p,t)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.V)*Central
end

# User definition of the set_parameters! function
function set_parameters(θ,η,z)
  @NT(Ka = z[:ka], CL = z[:cl], V = z[:v])
end
prob = ODEProblem(depot_model,zeros(2),(0.0,300.0))
pkpd = PKPDModel(prob,set_parameters)

# Simulate individual 1
η1 = zeros(2); θ = zeros(1)

VarType = promote_type(eltype(η1),eltype(θ))
p = set_parameters(θ,η1,data[1].z)
u0 = VarType.(pkpd.prob.u0)
tspan = VarType.(pkpd.prob.tspan)
@inferred ith_patient_cb(p,data[1],u0,tspan[1],prob)
tstops,cb = ith_patient_cb(p,data[1],u0,tspan[1],prob)
true_f = PKPDSimulator.DiffEqWrapper(pkpd.prob)
# Match the type of ηi for duality in estimator
prob = ODEProblem(true_f,u0,tspan,p,callback=cb)
save_start = true#datai.events[1].ss == 1
sol = solve(prob,Tsit5();save_start=save_start,tstops=tstops)

function reduction(sol,p,datai)
  (sol(datai.obs.times;idxs=2)./p.V)::Vector{typeof(sol.u[1][2])}
end
@inferred reduction(sol,p,data[1])
