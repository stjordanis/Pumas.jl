function simulate(_prob::DEProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> sol,
                  alg = Tsit5();kwargs...)
  prob,tstops = build_pkpd_problem(_prob,set_parameters,θ,ηi,datai)
  save_start = true#datai.events[1].ss == 1
  sol = solve(prob,alg;save_start=save_start,tstops=tstops,kwargs...)
  output_reduction(sol,sol.prob.p,datai)
end

function build_pkpd_problem(_prob,set_parameters,θ,ηi,datai)
  # From problem_new_parameters but no callbacks
  VarType = promote_type(eltype(ηi),eltype(θ))
  p = set_parameters(θ,ηi,datai.z)
  u0 = VarType.(_prob.u0)
  tspan = VarType.(_prob.tspan)
  tstops,cb = ith_patient_cb(p,datai,u0,tspan[1],_prob)
  true_f = DiffEqWrapper(_prob)
  problem_final_dispatch(_prob,true_f,u0,tspan,p,cb),tstops
end

function build_pkpd_problem(_prob::AbstractJumpProblem,set_parameters,θ,ηi,datai)
  prob,tstops = build_pkpd_problem(_prob.prob,set_parameters,θ,ηi,datai)
  JumpProblem{typeof(prob),typeof(_prob.aggregator),
              typeof(_prob.jump_callback),
              typeof(_prob.discrete_jump_aggregation),
              typeof(_prob.variable_jumps)}(prob,_prob.aggregator,
              _prob.discrete_jump_aggregation,
              _prob.jump_callback,_prob.variable_jumps),tstops
end

# Have separate dispatches to pass along extra pieces of the problem
function problem_final_dispatch(prob::ODEProblem,true_f,u0,tspan,p,cb)
  ODEProblem{DiffEqBase.isinplace(prob)}(true_f,u0,tspan,p,callback=cb)
end

function problem_final_dispatch(prob::SDEProblem,true_f,u0,tspan,p,cb)
  SDEProblem{DiffEqBase.isinplace(prob)}(true_f,prob.g,u0,tspan,p,callback=cb)
end

function problem_final_dispatch(prob::DDEProblem,true_f,u0,tspan,p,cb)
  DDEProblem{DiffEqBase.isinplace(prob)}(true_f,prob.h,u0,tspan,prob.p,
                   constant_lags = prob.constant_lags,
                   dependent_lags = prob.dependent_lags,
                   callback=cb)
end

function ith_patient_cb(p,datai,u0,t0,_prob)
  events = datai.events
  ss_abstol = 1e-12 # TODO: Make an option
  ss_reltol = 1e-12 # TODO: Make an option
  ss_max_iters = 1000

  lags,bioav,rate,duration = get_magic_args(p,u0,t0)
  adjust_event_timings!(events,lags,bioav,rate,duration)

  tstops = sorted_approx_unique(events)
  counter::Int = 1
  ss_mode = Ref(false)
  ss_time = Ref(-one(eltype(tstops)))
  ss_end = Ref(-one(eltype(tstops)))
  ss_rate_end = Ref(-one(eltype(tstops)))
  ss_cache = similar(u0) # TODO: Make this handle static arrays better.
  ss_ii = Ref(-one(eltype(tstops)))
  ss_duration = Ref(-one(eltype(tstops)))
  ss_overlap_duration = Ref(-one(eltype(tstops)))
  post_steady_state = Ref(false)
  ss_counter = Ref(0)
  ss_event_counter = Ref(0)
  ss_rate_multiplier = Ref(0)
  ss_dropoff_counter = Ref(0)
  ss_tstop_cache = Vector{eltype(tstops)}(0)
  last_restart = Ref(-one(eltype(tstops)))

  # searchsorted is empty iff t ∉ target_time
  # this is a fast way since target_time is sorted
  function condition(u,t,integrator)
    (post_steady_state[] && t == (ss_time[] + ss_overlap_duration[] + ss_dropoff_counter[]*ss_ii[])) ||
    t == ss_rate_end[] || (ss_mode[] && t == ss_end[] || !isempty(searchsorted(tstops,t)))
  end

  function affect!(integrator)

    if typeof(_prob) <: DDEProblem
      f = integrator.integrator.f
    else
      f = integrator.f
    end

    while counter <= length(events) && events[counter].time <= integrator.t
      cur_ev = events[counter]
      @inbounds if (cur_ev.evid == 1 || cur_ev.evid == -1) && cur_ev.ss == 0
        savevalues!(integrator)
        dose!(integrator,integrator.u,cur_ev,bioav,last_restart)
        counter += 1
      elseif cur_ev.evid >= 3
        savevalues!(integrator)

        if typeof(integrator.u) <: Union{Number,SArray}
          integrator.u = zero(integrator.u)
        else
          integrator.u .= 0
        end

        last_restart[] = integrator.t
        f.rates_on = false

        if typeof(f.rates) <: Union{Number,SArray}
          f.rates = zero(f.rates)
        else
          f.rates .= 0
        end

        cur_ev.evid == 4 && dose!(integrator,integrator.u,cur_ev,bioav,last_restart)
        counter += 1
      elseif cur_ev.ss > 0
        if !ss_mode[]
          savevalues!(integrator)
          # This is triggered at the start of a steady-state event
          ss_mode[] = true

          if typeof(f.rates) <: Union{Number,SArray}
            f.rates = zero(f.rates)
          else
            f.rates .= 0
          end

          ss_counter[] = 0
          integrator.opts.save_everystep = false
          post_steady_state[] = false
          typeof(_prob) <: SDEProblem && (integrator.W.save_everystep=false)

          ss_time[] = integrator.t
          # TODO: Handle saveat in this range
          # TODO: Make compatible with save_everystep = false
          if typeof(bioav) <: Number
            _duration = (bioav*cur_ev.amt)/cur_ev.rate
          else
            _duration = (bioav[cur_ev.amt]*cur_ev.amt)/cur_ev.rate
          end
          ss_duration[] = _duration
          ss_overlap_duration[] = mod(_duration,cur_ev.ii)
          ss_ii[] = cur_ev.ii
          ss_end[] = integrator.t + cur_ev.ii
          cur_ev.rate > 0 && (ss_rate_multiplier[] = 1 + (_duration>cur_ev.ii)*(_duration ÷ cur_ev.ii))
          ss_rate_end[] = integrator.t + ss_overlap_duration[]
          ss_cache .= integrator.u
          ss_dose!(integrator,integrator.u,cur_ev,bioav,ss_rate_multiplier,ss_rate_end)
          add_tstop!(integrator,ss_end[])
          cur_ev.rate > 0 && add_tstop!(integrator,ss_rate_end[])
        elseif integrator.t == ss_end[]
          integrator.t = ss_time[]
          typeof(_prob) <: SDEProblem && (integrator.W.curt = integrator.t)
          typeof(_prob) <: DDEProblem && (integrator.integrator.t = integrator.t)
          ss_cache .-= integrator.u
          err = integrator.opts.internalnorm(ss_cache)
          if ss_counter[] == ss_max_iters || (err < ss_abstol &&
             err/integrator.opts.internalnorm(integrator.u) < ss_reltol)
            # Steady state complete
            ss_mode[] = false
            # TODO: Make compatible with save_everystep = false
            post_steady_state[] = true

            if typeof(f.rates) <: Union{Number,SArray}
              f.rates = zero(f.rates)
            else
              f.rates .= 0
            end

            integrator.opts.save_everystep = true
            typeof(_prob) <: SDEProblem && (integrator.W.save_everystep=true)

            if cur_ev.ss == 2
              if typeof(integrator.u) <: Union{SArray,Number}
                integrator.u += integrator.sol.u[end]
              else
                integrator.u .+= integrator.sol.u[end]
              end
            end

            ss_dose!(integrator,integrator.u,cur_ev,bioav,ss_rate_multiplier,ss_rate_end)
            if cur_ev.rate > 0 && cur_ev.amt != 0 # amt = 0 means it never turns off
              ss_duration[] == cur_ev.ii ? start_k = 1 : start_k = 0
              for k in start_k:ss_rate_multiplier[]-1
                add_tstop!(integrator,integrator.t + ss_overlap_duration[] + k*ss_ii[])
              end
              ss_dropoff_counter[] = start_k
            else # amt = 0, turn off rates

              if typeof(f.rates) <: Union{Number,SArray}
                f.rates = StaticArrays.setindex(f.rates,0.0,cur_ev.cmt)
              else
                f.rates[cur_ev.cmt] .= 0
              end

              f.rates_on = false
            end
            if !isempty(ss_tstop_cache)
              # Put these tstops back in since they were erased in the ss interval
              for t in ss_tstop_cache
                add_tstop!(integrator,t)
              end
              resize!(ss_tstop_cache,0)
            end
            ss_event_counter[] = counter
            counter += 1
            post_steady_state[] = true
          else #Failure to converge, go again
            ss_cache .= integrator.u
            ss_counter[] += 1
            ss_dose!(integrator,integrator.u,cur_ev,bioav,ss_rate_multiplier,ss_rate_end)
            ss_rate_end[] < ss_end[] && add_tstop!(integrator,ss_rate_end[])
            #add_tstop!(integrator,ss_end[])
          end
        elseif integrator.t == ss_rate_end[] && cur_ev.amt != 0 # amt==0 means constant
          if typeof(f.rates) <: SArray
            f.rates = StaticArrays.setindex(f.rates,f.rates[cur_ev.cmt]-cur_ev.rate,cur_ev.cmt)
          else
            f.rates[cur_ev.cmt] -= cur_ev.rate
          end
          f.rates_on = (ss_rate_multiplier[] > 1)
        else
          # This is a tstop that was accidentally picked up in the interval
          # Just re-add so it's not missed after the SS
          integrator.t ∉ ss_tstop_cache && push!(ss_tstop_cache,integrator.t)
        end
        break # break when in SS because the counter doesn't increase
      elseif cur_ev.evid == 2
        #ignore for now
        counter += 1
      end
    end

    # Not at an event but still a tstop, turn off rates from last ss event
    if post_steady_state[] && integrator.t == ss_time[] + ss_overlap_duration[] + ss_dropoff_counter[]*ss_ii[]
      ss_dropoff_counter[] += 1

      # > is required if ss_overlap_duration[] == 0 since it starts at 1 to not
      # trigger at ss_time, but still needs to turn off!
      ss_dropoff_counter[] >= ss_rate_multiplier[] && (post_steady_state[] = false)


      ss_event = events[ss_event_counter[]]
      if ss_event.amt != 0
        if typeof(f.rates) <: SArray
          f.rates = StaticArrays.setindex(f.rates,f.rates[ss_event.cmt]-ss_event.rate,ss_event.cmt)
        else
          f.rates[ss_event.cmt] -= ss_event.rate
        end
      end
      # TODO: Optimize by setting f.rates_on = false
    end
  end
  tstops,DiscreteCallback{typeof(condition),typeof(affect!),
                          typeof(patient_cb_initialize!)}(condition,
                          affect!,patient_cb_initialize!,(false,false))
end

function dose!(integrator,u,cur_ev,bioav,last_restart)

  if typeof(integrator.sol.prob) <: DDEProblem
    f = integrator.integrator.f
  else
    f = integrator.f
  end

  if cur_ev.rate == 0
    if typeof(bioav) <: Number
      @views @. integrator.u[cur_ev.cmt] += bioav*cur_ev.amt
    else
      @views @. integrator.u[cur_ev.cmt] += bioav[cur_ev.cmt]*cur_ev.amt
    end
    savevalues!(integrator)
  else
    if cur_ev.rate_dir > 0 || integrator.t - cur_ev.duration > last_restart[]
      f.rates_on += cur_ev.evid > 0
      @views @. f.rates[cur_ev.cmt] += cur_ev.rate*cur_ev.rate_dir
    end
  end
end

function dose!(integrator,u::SArray,cur_ev,bioav,last_restart)

  if typeof(integrator.sol.prob) <: DDEProblem
    f = integrator.integrator.f
  else
    f = integrator.f
  end

  if cur_ev.rate == 0
    if typeof(bioav) <: Number
      integrator.u = StaticArrays.setindex(integrator.u,integrator.u[cur_ev.cmt] + bioav*cur_ev.amt,cur_ev.cmt)
    else
      integrator.u = StaticArrays.setindex(integrator.u,integrator.u[cur_ev.cmt] + bioav[cur_ev.cmt]*cur_ev.amt,cur_ev.cmt)
    end
    savevalues!(integrator)
  else
    if cur_ev.rate_dir > 0 || integrator.t - cur_ev.duration > last_restart[]
      f.rates_on += cur_ev.evid > 0
      f.rates = StaticArrays.setindex(f.rates,f.rates[cur_ev.cmt] + cur_ev.rate*cur_ev.rate_dir,cur_ev.cmt)
    end
  end
end

function ss_dose!(integrator,u,cur_ev,bioav,ss_rate_multiplier,ss_rate_end)

  if typeof(integrator.sol.prob) <: DDEProblem
    f = integrator.integrator.f
  else
    f = integrator.f
  end

  if cur_ev.rate > 0
    f.rates_on = true
    f.rates[cur_ev.cmt] = ss_rate_multiplier[]*cur_ev.rate*cur_ev.rate_dir
  else
    if typeof(bioav) <: Number
      integrator.u[cur_ev.cmt] += bioav*cur_ev.amt
    else
      integrator.u[cur_ev.cmt] += bioav[cur_ev.cmt]*cur_ev.amt
    end
  end
end

function ss_dose!(integrator,u::SArray,cur_ev,bioav,ss_rate_multiplier,ss_rate_end)

  if typeof(integrator.sol.prob) <: DDEProblem
    f = integrator.integrator.f
  else
    f = integrator.f
  end

  if cur_ev.rate > 0
    f.rates_on = true
    f.rates = StaticArrays.setindex(f.rates,ss_rate_multiplier[]*cur_ev.rate*cur_ev.rate_dir,cur_ev.cmt)
  else
    if typeof(bioav) <: Number
      integrator.u = StaticArrays.setindex(integrator.u,integrator.u[cur_ev.cmt]+bioav*cur_ev.amt,cur_ev.cmt)
    else
      integrator.u = StaticArrays.setindex(integrator.u,integrator.u[cur_ev.cmt]+bioav[cur_ev.cmt]*cur_ev.amt,cur_ev.cmt)
    end
  end
end


function patient_cb_initialize!(cb,t,u,integrator)
  if cb.condition(t,u,integrator)
    cb.affect!(integrator)
    u_modified!(integrator,true)
  else
    u_modified!(integrator,false)
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

mutable struct DiffEqWrapper{F,rateType} <: Function
  f::F
  rates_on::Int
  rates::rateType
end
function (f::DiffEqWrapper)(u,p,t)
  out = f.f(u,p,t)
  if f.rates_on > 0
    return out + rates
  else
    return out
  end
end
function (f::DiffEqWrapper)(du::T,u::T,p,t) where T
  f.f(du,u,p,t)
  f.rates_on > 0 && (du .+= f.rates)
end
function (f::DiffEqWrapper)(u::T,h,p,t) where T # h is DelayDiffEq.HistoryFunction
  f.f(du,u,h,p,t)
  if f.rates_on > 0
    return out + rates
  else
    return out
  end
end
function (f::DiffEqWrapper)(du::T,u::T,h,p,t) where T # h is DelayDiffEq.HistoryFunction
  f.f(du,u,h,p,t)
  f.rates_on > 0 && (du .+= f.rates)
end
DiffEqWrapper(prob) = DiffEqWrapper(prob.f,0,zeros(prob.u0))
DiffEqWrapper(f::DiffEqWrapper) = DiffEqWrapper(f.f,0,f.rates)
