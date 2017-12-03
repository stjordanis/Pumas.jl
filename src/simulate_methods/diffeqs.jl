function simulate(_prob::ODEProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> sol,
                  alg = Tsit5();kwargs...)
  VarType = promote_type(eltype(ηi),eltype(θ))
  p = set_parameters(θ,ηi,datai.z)
  target_time,cb = ith_patient_cb(p,datai,_prob)
  tstops = [_prob.tspan[1];target_time]
  # From problem_new_parameters but no callbacks

  true_f = DiffEqWrapper(_prob,p)
  # Match the type of ηi for duality in estimator
  prob = ODEProblem(true_f,VarType.(_prob.u0),VarType.(_prob.tspan),callback=cb)
  sol = solve(prob,alg;save_start=datai.events[1].rate != 0,tstops=tstops,kwargs...)
  output_reduction(sol,sol.prob.f.params,datai)
end

function ith_patient_cb(p,datai,prob)

  ss_tol = 1e-16 # TODO: Make an option
  ss_max_iters = Inf

  if haskey(p,:bioav)
    bioav = p.bioav
  else
    bioav = 1
  end

  target_time,events,tstop_times = adjust_event_timings(datai,p,bioav)

  counter = 1
  steady_state_mode = Ref(false)
  steady_state_end = Ref(-one(eltype(tstop_times)))
  steady_state_rate_end = Ref(-one(eltype(tstop_times)))
  steady_state_cache = similar(prob.u0)
  post_steady_state = Ref(false)
  ss_counter = Ref(0)

  # searchsorted is empty iff t ∉ target_time
  # this is a fast way since target_time is sorted
  condition = function (t,u,integrator)
    t == steady_state_rate_end[] || (steady_state_mode[] ? t == steady_state_end[] : !isempty(searchsorted(tstop_times,t)))
  end

  function affect!(integrator)
    while counter <= length(target_time) && target_time[counter].time <= integrator.t
      cur_ev = events[counter]
      @inbounds if (cur_ev.evid == 1 || cur_ev.evid == -1) && cur_ev.ss == 0
        savevalues!(integrator)
        if cur_ev.rate == 0
          if typeof(bioav) <: Number
            integrator.u[cur_ev.cmt] += bioav*cur_ev.amt
          else
            integrator.u[cur_ev.cmt] += bioav[cur_ev.cmt]*cur_ev.amt
          end
          savevalues!(integrator)
        else
          integrator.f.rates_on[] += cur_ev.evid > 0
          integrator.f.rates[cur_ev.cmt] += cur_ev.rate
        end
        counter += 1
      elseif cur_ev.ss == 1
        if !steady_state_mode[]
          savevalues!(integrator)
          # This is triggered at the start of a steady-state event
          steady_state_mode[] = true
          integrator.f.rates .= 0
          ss_counter[] = 0
          integrator.opts.save_everystep = false
          post_steady_state[] = false
          # TODO: Handle saveat in this range
          # TODO: Make compatible with save_everystep = false
          if typeof(bioav) <: Number
            steady_state_rate_end[] = integrator.t + (bioav*cur_ev.amt)/cur_ev.rate
          else
            steady_state_rate_end[] = integrator.t + (bioav[cur_ev.amt]*cur_ev.amt)/cur_ev.rate
          end
          steady_state_end[] = integrator.t + cur_ev.ii
          steady_state_cache .= integrator.u
          steady_state_dose(integrator,cur_ev,bioav,steady_state_rate_end)
          add_tstop!(integrator,steady_state_end[])
        elseif integrator.t == steady_state_end[]
          # TODO: Make non-allocating
          integrator.t = integrator.sol.t[end]
          if ss_counter[] == ss_max_iters || integrator.opts.internalnorm(integrator.u - steady_state_cache) < ss_tol
            # Steady state complete
            steady_state_mode[] = false
            # TODO: Make compatible with save_everystep = false
            integrator.f.rates .= 0
            integrator.opts.save_everystep = true
            steady_state_dose(integrator,cur_ev,bioav,steady_state_rate_end)
            counter += 1
            post_steady_state[] = true
          else
            steady_state_cache .= integrator.u
            ss_counter[] += 1
            steady_state_dose(integrator,cur_ev,bioav,steady_state_rate_end)
            add_tstop!(integrator,steady_state_end[])
          end
        elseif integrator.t == steady_state_rate_end[]
          integrator.f.rates .= 0
        end
        break
      end
    end
    if post_steady_state[] && integrator.t == steady_state_rate_end[]
      ss_event = events[counter-1]
      integrator.f.rates[ss_event.cmt] -= ss_event.rate
      post_steady_state[] = false
    end
  end
  tstop_times,DiscreteCallback(condition, affect!, initialize = patient_cb_initialize!,
                               save_positions=(false,false))
end

function steady_state_dose(integrator,cur_ev,bioav,steady_state_rate_end)
  if cur_ev.rate != 0
    integrator.f.rates_on[] = true
    integrator.f.rates[cur_ev.cmt] += cur_ev.rate
    add_tstop!(integrator,steady_state_rate_end[])
  else
    if typeof(bioav) <: Number
      integrator.u[cur_ev.cmt] += bioav*cur_ev.amt
    else
      integrator.u[cur_ev.cmt] += bioav[cur_ev.cmt]*cur_ev.amt
    end
  end
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

struct DiffEqWrapper{F,P,rateType} <: Function
  f::F
  params::P
  rates_on::Ref{Int}
  rates::Vector{rateType}
end
function (f::DiffEqWrapper)(t,u)
  out = f.f(t,u,f.params)
  if f.rates_on[] > 0
    return out + rates
  else
    return out
  end
end
function (f::DiffEqWrapper)(t,u,du)
  f.f(t,u,f.params,du)
  f.rates_on[] > 0 && (du .+= f.rates)
end
DiffEqWrapper(prob,p) = DiffEqWrapper(prob.f,p,Ref(0),zeros(prob.u0))
DiffEqWrapper(f::DiffEqWrapper,p) = DiffEqWrapper(f.f,p,Ref(0),f.rates)
