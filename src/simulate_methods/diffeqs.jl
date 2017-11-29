function simulate(_prob::ODEProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> (sol,false),
                  alg = Tsit5();kwargs...)
  p = set_parameters(θ,ηi,datai.z)
  target_time,cb = ith_patient_cb(p,datai)
  tstops = [_prob.tspan[1];target_time]
  # From problem_new_parameters but no callbacks

  true_f = DiffEqWrapper(_prob,p)
  prob = ODEProblem(true_f,_prob.u0,_prob.tspan,callback=cb)
  sol = solve(prob,alg;save_start=false,tstops=tstops,kwargs...)
  soli = first(output_reduction(sol,sol.prob.f.params,datai))
end

function ith_patient_cb(p,datai)

    if haskey(p,:bioav)
      bioav = p.bioav
    else
      bioav = 1
    end

    if !haskey(p,:lags)
      target_time = datai.event_times
      change_duration_by_bioav!(target_time,bioav)
      events = datai.events
      if bioav != 1
        order = sortperm(target_time)
        permute!(target_time,order)
        permute!(events,order)
      end
    else
      target_time,events = remove_lags(datai.events,datai.event_times,p.lags,bioav)
    end

    tstop_times = sorted_approx_unique(target_time)

    # searchsorted is empty iff t ∉ target_time
    # this is a fast way since target_time is sorted
    condition = function (t,u,integrator)
      !isempty(searchsorted(tstop_times,t))
    end
    counter = 1
    function affect!(integrator)
      while counter <= length(target_time) && target_time[counter].time <= integrator.t
        cur_ev = events[counter]
        @inbounds if (cur_ev.evid == 1 || cur_ev.evid == -1) && cur_ev.ss == 0
          if cur_ev.rate == 0
            if typeof(bioav) <: Number
              integrator.u[cur_ev.cmt] = bioav*cur_ev.amt
            else
              integrator.u[cur_ev.cmt] = bioav[cur_ev.cmt]*cur_ev.amt
            end
          else
            integrator.f.rates_on[] += cur_ev.evid > 0
            integrator.f.rates[cur_ev.cmt] += cur_ev.rate
          end
        elseif cur_ev.ss == 1
          #@show "here"
        end
        counter += 1
      end
    end

    tstop_times,DiscreteCallback(condition, affect!, initialize = patient_cb_initialize!)
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
