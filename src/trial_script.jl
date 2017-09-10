using DifferentialEquations
using DataFrames
using Distributions
cd(Pkg.dir("PKPDSimulator"))
using Plots; pyplot()

##########################################
# Load data

z = readtable("examples/oral1_1cpt_KAVCL_MD_data.txt", separator=' ')

##########################################
# Internal Functions

const p = zeros(3) # pre-cache p

function generate_η(ω,N)
  [rand(MvNormal(zeros(size(ω,1)),ω)) for i in 1:N]
end

function generate_individual_sol(prob,η,z,i)
 set_parameters!(p,prob.u0,η,z,i) # from the user
 set_param_values!(prob.f,p) # this is in DiffEqBase: sets values in f
 sol = solve(prob,Tsit5()) # solve the diffeq, return the solution
end

function simulate(prob,ω,z,alg=Tsit5(),ϵ=nothing;kwargs...)
  N = maximum(z[:id])
  η = generate_η(ω,N)
  prob_func = function (prob,i,repeat)
    set_parameters!(p,prob.u0,η[i],z,i) # from the user
    set_param_values!(prob.f,p) # this is in DiffEqBase: sets values in f
    prob.callback = nth_patient_cb(i)
    prob
  end
  monte_prob = MonteCarloProblem(prob,prob_func=prob_func)
  sol = solve(monte_prob,alg;parallel_type=:none,num_monte=N,kwargs...)
  ϵ != nothing && add_noise!(sol,ϵ)
  sol
end

#########################################
# User Stuff
#########################################
# User definition of the ODE Function

# Option 1: The macro
f = @ode_def_nohes DepotModel begin
  dDepot = -Ka*Depot
  dCentral = Ka*Depot - (CL/V)*Central
end Ka=>2.0 CL=>20.0 V=>100.0

# Option 2: The function
function depot_model(t,u,p,du)
 Depot,Central = u
 Ka,CL,V = p
 du[1] = -Ka*Depot
 du[2] =  Ka*Depot - (CL/V)*Central
end
f = ParameterizedFunction(depot_model,[2.0,20.0,100.0])

###########################################
# User definition of the set_parameters! function

function set_parameters!(p,u0,η,z,i)
  col = getθ!(p, z, i)
  p[2] *= exp(η[1])
  p[3] *= exp(η[2])
  u0[1] = float(first(z[col, :][:amt]))
end

function getθ!(dest,z,i)
    col = findfirst(x->x==i, z[:id])
    d_n = z[col,:]
    for (i,para) ∈ enumerate((:ka, :cl, :v))
        dest[i] = float(first(d_n[para]))
    end
    col
end

#############################################
# The DiscreteCallback

function get_tstops(z)
    d_1 = nth_patient(z,1)
    cols = find(x->x==1, d_1[:evid])
    target_time = d_1[:time][cols]
end

function nth_patient_cb(n)
    d_n = nth_patient(z,n)
    cols = find(x->x==1, d_n[:evid])
    target_time = d_n[:time][cols]
    condition = (t,u,integrator) -> t ∈ target_time
    counter = 1
    function affect!(integrator)
        integrator.u[1] = d_n[:amt][cols][counter]
        counter += 1
    end
    DiscreteCallback(condition, affect!)
end

############################################
# Get N-th patient's data

function nth_patient(p_data,n)
  n_start = findfirst(x -> x==n, p_data[:id])
  n_end   = findlast(x ->  x==n, p_data[:id])
  p_data[n_start:n_end, :]
end

## Finish the ODE Definition

tspan = (0.0,300.0)
u0 = zeros(2)
prob = ODEProblem(f,u0,tspan,callback=nth_patient_cb(1))
ts = get_tstops(z)

############################################
# Population setup

# θ = [2.268,74.17,468.6,0.5876] # [2.0,3.0,10.0,1.0] # Unfitted
ω = [0.05 0.0
     0.0 0.2]

#############################################
# Call simulate

sol = simulate(prob,ω,z;tstops=ts)
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.1:19)
plot(summ,title="Summary plot",xlabel="time")
