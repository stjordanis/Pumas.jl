function simulate(prob::AnalyticalProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> (sol,false);
                  parallel_type=:threads,kwargs...)
  VarType = promote_type(eltype(ηi),eltype(θ))
  u0 = VarType.(prob.u0)
  f = prob.f
  t0 = datai.event_times[1].time
  tspan = VarType.(prob.tspan)
  tdir = sign(prob.tspan[end] - prob.tspan[1])
  tstops = [tspan[1];datai.event_times] # Doesn't make sure it stops at tspan[end]!
  p = set_parameters(θ,ηi,datai.z)
  u_save = Vector{typeof(u0)}(length(tstops))
  u_save[1] = u0
  for i in 2:length(tstops)
    t = tstops[i]
    cur_ev = datai.events[i-1]
    dose = create_dose_vector(cur_ev,u0)
    u0 = f(t,t0,u0,dose,p)
    u_save[i] = u0
    t0 = t
  end
  _soli = PKPDAnalyticalSolution{typeof(u0),ndims(u0)+1,typeof(u_save),
                     typeof(tstops),
                     typeof(p),
                     typeof(prob),typeof(datai.events)}(
                     u_save,tstops,p,prob,datai.events,true,0,:Success)
  output_reduction(_soli,p,datai)
end

function create_dose_vector(cur_ev,u0)
  increment_value(zero(u0),cur_ev.amt,cur_ev.cmt)
end

function increment_value(A::SVector{L,T},x,k) where {L,T}
    [ifelse(i == k, A[i]+x, A[i]) for i in 1:L]
end
