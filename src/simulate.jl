function generate_η(ω,N)
  [rand(MvNormal(zeros(size(ω,1)),ω)) for i in 1:N]
end

function simulate(f,tspan,num_dependent,set_parameters!,θ,ω,z,alg=Tsit5(),ϵ=nothing;kwargs...)
  u0 = zeros(num_dependent)
  tstops = [tspan[1];z[1].event_times] # assumes all of the event times are the same!
  tspan = (tspan[1]-1e-12,tspan[2]) # for initial condition hack, see #7
  prob = ODEProblem(f,u0,tspan,callback=ith_patient_cb(z,1))
  N = length(z)
  η = generate_η(ω,N)
  prob_func = function (prob,i,repeat)
    p = zeros(num_params(prob.f))
    set_parameters!(p,prob.u0,θ,η[i],z[i]) # from the user
    set_param_values!(prob.f,p) # this is in DiffEqBase: sets values in f
    prob.callback = ith_patient_cb(z,i)
    prob
  end
  monte_prob = MonteCarloProblem(prob,prob_func=prob_func)
  sol = solve(monte_prob,alg;parallel_type=:none,num_monte=N,
              save_start=false,tstops=tstops,kwargs...)
  ϵ != nothing && add_noise!(sol,ϵ)
  sol
end

function ith_patient_cb(z,i)
    d_n = z[i].events
    target_time = z[i].event_times
    condition = (t,u,integrator) -> t ∈ target_time
    counter = 1
    function affect!(integrator)
        integrator.u[1] = z[i].events[counter,:amt]
        counter += 1
    end
    DiscreteCallback(condition, affect!)
end
