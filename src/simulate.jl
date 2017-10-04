function generate_η(ω,N)
  [rand(MvNormal(zeros(size(ω,1)),ω)) for i in 1:N]
end

function simulate(f,tspan,num_dependent,set_parameters!,θ,ω,data,
                  output_func=(sol,i)-> (sol,false),alg=Tsit5(),
                  ϵ=nothing;parallel_type=:threads,kwargs...)
  u0 = zeros(num_dependent)
  tstops = [tspan[1];get_all_event_times(data)] # uses tstops on all, could be by individual
  tspan = (tspan[1]-1e-12,tspan[2]) # for initial condition hack, see #7
  prob = ODEProblem(f,u0,tspan,callback=ith_patient_cb(data,1))
  N = length(data)
  η = generate_η(ω,N)
  prob_func = function (prob,i,repeat)
    p = zeros(num_params(prob.f))
    set_parameters!(p,θ,η[i],data[i].z)
    set_param_values!(prob.f,p) # this is in DiffEqBase: sets values in f
    prob.callback = ith_patient_cb(data,i)
    prob
  end
  monte_prob = MonteCarloProblem(prob,prob_func=prob_func,output_func=output_func)
  sol = solve(monte_prob,alg;num_monte=N,save_start=false,
              tstops=tstops,kwargs...)
  ϵ != nothing && add_noise!(sol,ϵ)
  sol
end

function ith_patient_cb(data,i)
    d_n = data[i].events
    target_time = data[i].event_times
    condition = (t,u,integrator) -> t ∈ target_time
    counter = 1
    function affect!(integrator)
      cur_ev = data[i].events[counter]
      integrator.u[cur_ev.cmt] = cur_ev.amt
      counter += 1
    end
    DiscreteCallback(condition, affect!)
end

function get_all_event_times(data)
  total_times = copy(data[1].event_times)
  for i in 2:length(data)
    for t in data[i].event_times
      t ∉ total_times && push!(total_times,t)
    end
  end
  total_times
end
