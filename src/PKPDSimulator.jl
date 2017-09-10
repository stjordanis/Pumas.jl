module PKPDSimulator

using DiffEqBase, DiffEqMonteCarlo, Distributions, Reexport

@reexport using OrdinaryDiffEq

function generate_η(ω,N)
  [rand(MvNormal(zeros(size(ω,1)),ω)) for i in 1:N]
end

function generate_individual_sol(prob,η,z,i)
 set_parameters!(p,prob.u0,η,z,i) # from the user
 set_param_values!(prob.f,p) # this is in DiffEqBase: sets values in f
 sol = solve(prob,Tsit5()) # solve the diffeq, return the solution
end

function simulate(prob,set_parameters!,θ,ω,z,alg=Tsit5(),ϵ=nothing;kwargs...)
  N = maximum(z[:id])
  η = generate_η(ω,N)
  prob_func = function (prob,i,repeat)
    p = zeros(num_params(prob.f))
    set_parameters!(p,prob.u0,θ,η[i],z,i) # from the user
    set_param_values!(prob.f,p) # this is in DiffEqBase: sets values in f
    prob.callback = nth_patient_cb(z,i)
    prob
  end
  monte_prob = MonteCarloProblem(prob,prob_func=prob_func)
  sol = solve(monte_prob,alg;parallel_type=:none,num_monte=N,kwargs...)
  ϵ != nothing && add_noise!(sol,ϵ)
  sol
end

function nth_patient_cb(z,n)
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

export simulate, nth_patient_cb, nth_patient

end # module
