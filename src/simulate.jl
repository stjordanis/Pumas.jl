function generate_η(ω,N)
  [rand(MvNormal(zeros(size(ω,1)),ω)) for i in 1:N]
end

function simulate(_prob,set_parameters,θ,ω,data::Population,
                  output_reduction = (sol,p,datai) -> (sol,false),
                  ϵ = nothing, error_model = nothing,
                  alg = Tsit5();parallel_type=:threads,kwargs...)
  prob = ODEProblem(_prob.f,_prob.u0,_prob.tspan,callback=ith_patient_cb(data[1]))
  tstops = [prob.tspan[1];get_all_event_times(data)] # uses tstops on all, could be by individual
  N = length(data)
  η = generate_η(ω,N)
  prob_func = function (prob,i,repeat)
    # From problem_new_parameters but no callbacks
    f = ParameterizedFunction(prob.f,set_parameters(θ,η[i],data[i].z))
    uEltype = eltype(θ)
    u0 = [uEltype(prob.u0[i]) for i in 1:length(prob.u0)]
    tspan = (uEltype(prob.tspan[1]),uEltype(prob.tspan[2]))
    ODEProblem(f,u0,tspan,callback=ith_patient_cb(data[i]))
  end
  output_func = function (sol,i)
    output_reduction(sol,sol.prob.f.params,data[i])
  end
  monte_prob = MonteCarloProblem(prob,prob_func=prob_func,output_func=output_func)
  sol = solve(monte_prob,alg;num_monte=N,save_start=false,
              tstops=tstops,kwargs...)
  if error_model != nothing
    err_sol = [error_model(soli,η,rand(ϵ,length(soli))) for soli in sol]
  else
    err_sol = sol
  end
  err_sol
end

function simulate(_prob,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> (sol,false),
                  ϵ = nothing, error_model = nothing,
                  alg = Tsit5();parallel_type=:threads,kwargs...)
  tstops = [_prob.tspan[1];datai.obs_times]
  # From problem_new_parameters but no callbacks
  true_f = ParameterizedFunction(_prob.f,set_parameters(θ,ηi,datai.z))
  prob = ODEProblem(true_f,_prob.u0,_prob.tspan,callback=ith_patient_cb(datai))
  sol = solve(prob,alg;save_start=false,tstops=tstops,kwargs...)
  soli = first(output_reduction(sol,sol.prob.f.params,datai))
  if error_model != nothing
    _ϵ = rand(ϵ,length(soli))
    err_sol = error_model(soli,ηi,_ϵ)
  else
    err_sol = soli
  end
  err_sol
end

function ith_patient_cb(datai)
    d_n = datai.events
    target_time = datai.event_times
    condition = (t,u,integrator) -> t ∈ target_time
    counter = 1
    function affect!(integrator)
      cur_ev = datai.events[counter]
      integrator.u[cur_ev.cmt] = cur_ev.amt
      counter += 1
    end
    DiscreteCallback(condition, affect!, initialize = patient_cb_initialize!)
end

function patient_cb_initialize!(cb,t,u,integrator)
  if cb.condition(t,u,integrator)
    cb.affect!(integrator)
  end
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
