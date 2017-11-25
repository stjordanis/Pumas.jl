using PKPDSimulator, NamedTuples

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), covariates,dvs,
              separator=' ')

# Define the ODE
function depot_model(t,t0,u0,D,p)
  exp(-t*(p.Ka+p.CL/p.V))*(D*exp(p.Ka*t0+p.CL*t/p.V)*p.Ka*p.V +
  exp(t*p.Ka+p.CL*t0/p.V)*(u0*p.CL - (u0 + D)*p.Ka*p.V))/(p.CL-p.Ka*p.V)
end


# User definition of the set_parameters! function
function set_parameters(θ,η,z)
  @NT(Ka = z[:ka], CL = z[:cl], V = z[:v])
end

# Population setup

θ = zeros(1) # Not used in this case
ω = zeros(2)

# Call simulate
prob = AnalyticalProblem(depot_model,0.0,0.0,(0.0,300.0))
sol = simulate(prob,set_parameters,θ,ω,data)

# Simulate individual 1
η1 = zeros(2)
sol1 = simulate(prob,set_parameters,θ,η1,data[1])
