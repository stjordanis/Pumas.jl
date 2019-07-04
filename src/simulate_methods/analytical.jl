function _solve_analytical(m::PumasModel, subject::Subject, tspan, col,
                           args...; continuity = :right, kwargs...)
  f = m.prob isa ExplicitModel ? m.prob : m.prob.pkprob
  u0 = pk_init(f)

  # we don't want to promote units
  if numtype(col) <: Unitful.Quantity || numtype(u0) <: Unitful.Quantity || numtype(tspan) <: Unitful.Quantity
    Tu0 = map(float, u0)
    Ttspan = map(float, tspan)
  else
    T = promote_type(numtype(col), numtype(u0), numtype(tspan))
    Tu0 = convert.(T,u0)
    Ttspan = map(float, tspan)
  end

  prob = PKPDAnalyticalProblem{false}(f, Tu0, Ttspan)
  ss = prob.ss

  lags,bioav,time_adjust_rate,duration = get_magic_args(col,Tu0,Ttspan[1])
  events = adjust_event(subject.events,u0,lags,bioav,time_adjust_rate,duration)
  times = sorted_approx_unique(events)
  u = Vector{typeof(Tu0)}(undef, length(times))
  doses = Vector{typeof(Tu0)}(undef, length(times))
  rates = Vector{typeof(Tu0)}(undef, length(times))

  t0 = times[1]
  rate = zero(Tu0)
  last_dose = zero(Tu0)
  i = 1
  ss_time = -one(t0)
  ss_overlap_duration = -one(t0)
  ss_rate_multiplier = -1.0
  post_ss_counter = -1
  event_counter = i-1
  ss_rate = zero(rate)
  ss_ii = zero(t0)
  ss_cmt = 0
  ss_dropoff_event = false
  start_val = 0
  retcode = :Success
  local dose
  # Now loop through the rest
  while i <= length(times)
    t = times[i]
    ss_dropoff_event = post_ss_counter < ss_rate_multiplier + start_val &&
                       t == ss_time + ss_overlap_duration + post_ss_counter*ss_ii
    if ss_dropoff_event
      # Do an off event from the ss
      post_ss_counter += 1
      Tu0 = f(t,t0,Tu0,dose,col,rate)
      u[i] = Tu0
      doses[i] = zero(Tu0)
      _rate = create_ss_off_rate_vector(ss_rate,ss_cmt,rate)
      rate = _rate
      rates[i] = rate
    else # Not an SS off event
      ss_dropoff_event = false
      event_counter += 1
      cur_ev = events[event_counter]
      cur_ev.evid >= 3 && (Tu0 = zero(Tu0))
      @assert cur_ev.time == t
      if cur_ev.ss == 0
        dose,_rate = create_dose_rate_vector(cur_ev,Tu0,rate,bioav)

        (t0 != t) && cur_ev.evid < 3 && (Tu0 = f(t,t0,Tu0,last_dose,col,rate))
        rate = _rate
        u[i] = Tu0
        doses[i] = dose
        last_dose = dose
        rates[i] = rate
      else # handle steady state
        ss_time = t0
        # No steady state solution given, use a fallback
        dose,_rate = create_ss_dose_rate_vector(cur_ev,Tu0,rate,bioav)
        if cur_ev.rate > 0
          ss_rate = _rate
          _duration = (bioav*cur_ev.amt)/cur_ev.rate
          ss_overlap_duration = mod(_duration,cur_ev.ii)
          ss_rate_multiplier = 1 + (_duration>cur_ev.ii)*(_duration ÷ cur_ev.ii)
          ss_cmt = cur_ev.cmt
          ss_ii = cur_ev.ii
          _rate *= ss_rate_multiplier
          _t1 = t0 + ss_overlap_duration
          _t2 = t0 + cur_ev.ii
          _t1 == t0 && (_t1 = _t2)

          cur_ev.ss==2 && (Tu0_cache = f(t,t0,Tu0,last_dose,col,rate))

          if ss == nothing
            Tu0prev = Tu0 .+ 1
            while norm(Tu0-Tu0prev) > 1e-12
              Tu0prev = Tu0
              Tu0 = f(_t1,t0,Tu0,dose,col,_rate)
              _t2-_t1 > 0 && (Tu0 = f(_t2,_t1,Tu0,zero(Tu0),col,_rate-ss_rate))
            end
          end

          rate = _rate
          cur_ev.amt == 0 && (rate = zero(rate))
          cur_ev.ss == 2 && (Tu0 += Tu0_cache)

          u[i] = Tu0
          doses[i] = dose
          last_dose = dose
          rates[i] = rate
          if cur_ev.amt != 0
            old_length = length(times)
            ss_overlap_duration == 0 ? start_val = 1 : start_val = 0
            resize!(times,old_length+Int(ss_rate_multiplier))
            resize!(u,    old_length+Int(ss_rate_multiplier))
            resize!(doses,old_length+Int(ss_rate_multiplier))
            resize!(rates,old_length+Int(ss_rate_multiplier))
            for j in start_val:Int(ss_rate_multiplier)-1+start_val
              times[old_length+j+1-start_val] = ss_time + ss_overlap_duration + j*cur_ev.ii
            end
            post_ss_counter = start_val
            times = sort!(times)
          end
        else # Not a rate event, just a dose

          _t1 = t0 + cur_ev.ii
          cur_ev.ss==2 && (Tu0_cache = f(t,t0,Tu0,last_dose,col,rate))

          if ss == nothing
            Tu0prev = Tu0 .+ 1
            while norm(Tu0-Tu0prev) > 1e-12
              Tu0prev = Tu0
              Tu0 = f(_t1,t0,Tu0,dose,col,_rate)
            end
          end

          rate = _rate
          cur_ev.ss == 2 && (Tu0 += Tu0_cache)
          u[i] = Tu0
          doses[i] = dose
          last_dose = dose
          rates[i] = rate

        end
      end
    end

    if any(x->any(isnan,x),u[i])
      retcode = :Unstable
      break
    end

    # Don't update i for duplicate events
    t0 = t
    if ss_dropoff_event || (event_counter != length(events) && events[event_counter+1].time != t) || event_counter == length(events)
      i += 1
    end
  end

  # The two assumes that all equations are vector equations
  PKPDAnalyticalSolution{typeof(Tu0),
                         2,
                         typeof(u),
                         typeof(times),
                         typeof(doses),
                         typeof(rates),
                         typeof(col),
                         typeof(prob)}(u,times,doses,rates,col,prob,
                                       true,0,retcode,continuity)
end

function create_dose_rate_vector(cur_ev,u0,rate,bioav)
  if cur_ev.rate == 0
    if typeof(bioav) <: Number
      return increment_value(zero(u0),bioav*cur_ev.amt,cur_ev.cmt),rate
    else
      return increment_value(zero(u0),bioav[cur_ev.cmt]*cur_ev.amt,cur_ev.cmt),rate
    end
  else
    return zero(u0),increment_value(rate,cur_ev.rate_dir*cur_ev.rate,cur_ev.cmt)
  end
end

function create_ss_dose_rate_vector(cur_ev,u0,rate,bioav)
  if cur_ev.rate == 0
    if typeof(bioav) <: Number
      return set_value(zero(u0),bioav*cur_ev.amt,cur_ev.cmt),rate
    else
      return set_value(zero(u0),bioav[cur_ev.cmt]*cur_ev.amt,cur_ev.cmt),rate
    end
  else
    return zero(u0),set_value(zero(u0),cur_ev.rate_dir*cur_ev.rate,cur_ev.cmt)
  end
end

function create_ss_off_rate_vector(ss_rate,ss_cmt,rate)
  if rate == 0
    error("Too many off events, rate went negative!")
  else
    return increment_value(rate,-ss_rate[ss_cmt],ss_cmt)
  end
end
