function simulate(prob::AnalyticalProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> (sol,false);
                  parallel_type=:threads,kwargs...)
  u0 = prob.u0
  aux = prob.aux
  f = prob.f
  t0 = datai.event_times[1].time
  tspan = prob.tspan
  tdir = sign(prob.tspan[end] - prob.tspan[1])
  tstops = [tspan[1];datai.event_times] # Doesn't make sure it stops at tspan[end]!
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
  soli
end
