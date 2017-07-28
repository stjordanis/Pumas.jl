using DifferentialEquations

#########################################
# User Stuff
#########################################
# User definition of the ODE Function

f = @ode_def_nohes DepotModel begin
  dDepot = -Ka*Depot
  dCentral = Ka*Depot - (CL/V)*Central
end Ka=>2.0 CL=>20.0 V=>100.0

function depot_model(t,u,p,du)
 Depot,Central = u
 Ka,CL,V = p
 du[1] = -Ka*Depot
 du[2] =  Ka*Depot - (CL/V)*Central
end
f = ParameterizedFunction(depot_model,[2.0,20.0,100.0])

###########################################
# User definition of the parameter function

function set_parameters!(p,θ,η,z)
  wt,sex = z
  Ka = θ[1]
  CL = θ[2]*((wt/70)^0.75)*(θ[4]^sex)*exp(η[1])
  V  = θ[3]*exp(η[2])
  p[1] = Ka, p[2] = CL, p[3] = V
end

############################################
# Population setup

θ = [2.0,3.0,10.0,1.0]
ω = [0.05 0.0
     0.0 0.2]
z = # Dataset, Read csv into DataFrame

ϵ = # ?

simulate(θ,ω,z,ϵ)

##########################################
# Internal Functions

const p = zeros(3) # pre-cache p

function generate_η(ω)
  # Generate size(ω,1) x N matrix from ω, the varcovar matrix
end

function

function generate_individual_sol(θ,η,z,i)
 set_parameters!(p,θ,η,z,i) # from the user
 set_param_values!(prob.f,p) # this is in DiffEqBase: sets values in f
 sol = solve(prob,Tsit5()) # solve the diffeq, return the solution
end

function simulate(θ,ω,z,ϵ)
  η = generate_η(ω)
  N = maximum(z[:id])
  for i in N
    sols[i] = generate_individual_sol(θ,η[:,i],z,i)
  end
  add_noise!(sols,ϵ)
  sols
end
