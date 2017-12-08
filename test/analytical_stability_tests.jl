using PKPDSimulator, NamedTuples, Base.Test

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

# Simulate individual 1
η1 = zeros(2); θ = zeros(1)
@inferred simulate(pkpd.prob,pkpd.set_parameters,θ,η1,data[1],pkpd.reduction)

sol = simulate(pkpd.prob,pkpd.set_parameters,θ,η1,data[1],pkpd.reduction)
function reduction(sol,p,datai)
  sol(datai.obs_times;idxs=2)./p.V
end
pkpd = PKPDModel(OneCompartmentModel(320.0),set_parameters,reduction)
p = set_parameters(θ,η1,data[1].z)

# Fails because of idxs keyword argument. Should pass on v0.7/1.0?
@test_broken @inferred reduction(sol,p,data[1])
@test_broken @inferred simulate(pkpd.prob,pkpd.set_parameters,θ,η1,data[1],pkpd.reduction)

function reduction(sol,p,datai)
  (sol(datai.obs_times;idxs=2)./p.V)::Vector{typeof(sol.u[1][2])}
end
@inferred reduction(sol,p,data[1])
@inferred simulate(pkpd.prob,pkpd.set_parameters,θ,η1,data[1],pkpd.reduction)
