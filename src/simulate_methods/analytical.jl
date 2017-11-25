function simulate(prob::AnalyticalProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> (sol,false),
                  ϵ = nothing, error_model = nothing;
                  parallel_type=:threads,kwargs...)
  u0 = prob.u0
  aux = prob.aux
  f = prob.f
  t0 = datai.event_times[1]
  tspan = prob.tspan
  tdir = sign(prob.tspan[end] - prob.tspan[1])
  tstops = vec(collect(typeof(t0),Iterators.filter(
            x->tdir*tspan[1]<=tdir*x<=tdir*tspan[end],
            Iterators.flatten((datai.event_times,tspan[end])))))

  p = set_parameters(θ,ηi,datai.z)
  u_save = Vector{typeof(u0)}(length(tstops))
  aux_save = Vector{typeof(aux)}(length(tstops))
  u_save[1] = u0
  aux_save[1] = aux
  for i in 2:length(tstops)
    t = tstops[i]
    cur_ev = datai.events[i-1]
    u0,aux = f(t,t0,u0,cur_ev.amt,p,aux)
    u_save[i] = u0
    aux_save[i] = aux
    t0 = t
  end
  _soli = PKPDAnalyticalSolution{typeof(u0),ndims(u0)+1,typeof(u_save),
                     typeof(aux_save),
                     typeof(tstops),
                     typeof(p),
                     typeof(prob),typeof(datai.events)}(
                     u_save,aux_save,tstops,p,prob,datai.events,true,0,:Success)
  soli,_ = output_reduction(_soli,p,datai)
  if error_model != nothing
    _ϵ = rand(ϵ,length(soli))
    err_sol = error_model(soli,ηi,_ϵ)
  else
    err_sol = soli
  end
  err_sol
end

function simulate(prob::AnalyticalProblem,set_parameters,θ,ω,data::Population,
                  output_reduction = (sol,p,datai) -> (sol,false),
                  ϵ = nothing, error_model = nothing;
                  parallel_type=:threads,kwargs...)
    N = length(data)
    η = generate_η(ω,N)
    t = @elapsed u =
    [simulate(prob,set_parameters,θ,η[i],data[i],output_reduction,ϵ,error_model)
    for i in 1:length(data)]
    MonteCarloSolution(u,t,true)
end
