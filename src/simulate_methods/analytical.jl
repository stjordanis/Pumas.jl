function simulate(prob::AnalyticalProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> (sol,false);
                  parallel_type=:threads,kwargs...)
  VarType = promote_type(eltype(ηi),eltype(θ))
  u0 = VarType.(prob.u0)
  f = prob.f
  tspan = VarType.(prob.tspan)
  tdir = sign(prob.tspan[end] - prob.tspan[1])
  times = [t.time for t in datai.event_times]
  p = set_parameters(θ,ηi,datai.z)
  u = Vector{typeof(u0)}(length(times))
  doses = Vector{typeof(u0)}(length(times))
  rates = Vector{typeof(u0)}(length(times))

  # Iteration 1
  t0 = times[1]
  cur_ev = datai.events[1]
  u[1] = u0
  dose,rate = create_dose_rate_vector(cur_ev,u0,zero(u0))
  doses[1] = dose
  rates[1] = rate

  # Now loop through the rest
  for i in 2:length(times)
    t = times[i]
    cur_ev = datai.events[i]
    dose,_rate = create_dose_rate_vector(cur_ev,u0,rate)
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

function create_dose_rate_vector(cur_ev,u0,rate)
  if cur_ev.rate == 0
    return increment_value(zero(u0),cur_ev.amt,cur_ev.cmt),rate
  else
    return zero(u0),increment_value(rate,cur_ev.rate,cur_ev.cmt)
  end
end

function increment_value(A::SVector{L,T},x,k) where {L,T}
    _A = [i == k ? x : zero(x) for i in 1:L]
    A+_A
end

function increment_value(A::Number,x,k)
  A+x
end
