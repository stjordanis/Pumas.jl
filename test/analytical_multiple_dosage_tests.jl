using PKPDSimulator, NamedTuples

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),
              "examples/oral1_1cpt_KAVCL_MD_data.txt"), covariates,dvs)

# Define the ODE
prob = OneCompartmentModel(19.0)

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
