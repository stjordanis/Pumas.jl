function simulate(prob::PKPDAnalyticalProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> sol;
                  kwargs...)
  VarType = promote_type(eltype(ηi),eltype(θ))
  u0 = VarType.(prob.u0)
  f = prob.f
  tspan = VarType.(prob.tspan)
  tdir = sign(prob.tspan[end] - prob.tspan[1])
  p = set_parameters(θ,ηi,datai.z)
  lags,bioav,rate,duration = get_magic_args(p,u0,tspan[1])
  events = datai.events
  adjust_event_timings!(events,lags,bioav,rate,duration)
  times = sorted_approx_unique(events)
  u = Vector{typeof(u0)}(length(times))
  doses = Vector{typeof(u0)}(length(times))
  rates = Vector{typeof(u0)}(length(times))

  # Iteration 1
  t0 = times[1]
  cur_ev = events[1]
  u[1] = u0
  dose,rate = create_dose_rate_vector(cur_ev,u0,zero(u0),bioav)
  doses[1] = dose
  rates[1] = rate

  # Now loop through the rest
  # TODO: Handle multiple events at the same time
  for i in 2:length(times)
    t = times[i]
    cur_ev = events[i]
    dose,_rate = create_dose_rate_vector(cur_ev,u0,rate,bioav)
    u0 = f(t,t0,u0,dose,p,rate)
    rate = _rate
    u[i] = u0
    doses[i] = dose
    rates[i] = rate
    t0 = t
  end
  _soli = PKPDAnalyticalSolution{typeof(u0),ndims(u0)+1,typeof(u),
                     typeof(times),
                     typeof(doses),typeof(rates),
                     typeof(p),
                     typeof(prob)}(
                     u,times,doses,rates,p,prob,true,0,:Success)
  output_reduction(_soli,p,datai)
end

function create_dose_rate_vector(cur_ev,u0,rate,bioav)
  if cur_ev.rate == 0
    if typeof(bioav) <: Number
      return increment_value(zero(u0),bioav*cur_ev.amt,cur_ev.cmt),rate
    else
      return increment_value(zero(u0),bioav[cur_ev.cmt]*cur_ev.amt,cur_ev.cmt),rate
    end
  else
    return zero(u0),increment_value(rate,cur_ev.rate,cur_ev.cmt)
  end
end
