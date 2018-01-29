using PKPDSimulator, NamedTuples

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), covariates,dvs)

# Define the ODE
prob = OneCompartmentModel(320.0)

# User definition of the set_parameters! function
function set_parameters(θ,η,z)
  @NT(Ka = z[:ka], CL = z[:cl], V = z[:v])
end
pkpd = PKPDModel(prob,set_parameters)

# Population setup

θ = zeros(1) # Not used in this case
ω = zeros(2)

# Call simulate
sol = simulate(pkpd,θ,ω,data)

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(pkpd,θ,η1,data[1])


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

# Population setup

θ = zeros(1) # Not used in this case
ω = zeros(2)

# Call simulate
sol2 = simulate(pkpd,θ,ω,data[1],abstol=1e-14,reltol=1e-14)
@test norm(sol1(0.5:1:300-0.5) .- sol2(0.5:1:300-0.5).u) < 1e-6
