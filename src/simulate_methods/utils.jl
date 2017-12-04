function adjust_event_timings!(events,lags,bioav,rate,duration)
  for i in eachindex(events)
    ev = events[i]
    @assert rate == 0 || duration == 0

    if rate != 0
      _rate = rate
      _duration = ev.amt
      typeof(rate) <: Number ? (_duration /= rate) : (_duration /= rate[ev.cmt])
      typeof(bioav) <: Number ? (_duration *= bioav) : (_duration *= bioav[ev.cmt])
    elseif duration != 0
      typeof(duration) <: Number ? (_duration = duration) : (_duration = duration[ev.cmt])
      typeof(bioav) <: Number ? (_rate *= bioav) : (_rate *= bioav[ev.cmt])
    else # both are zero
      _rate = ev.rate
      _duration = ev.amt/ev.rate
      typeof(bioav) <: Number ? (_duration *= bioav) : (_duration *= bioav[ev.cmt])
    end

    if ev.off_event
      time = ev.base_time + _duration
    else
      time = ev.base_time
    end

    typeof(lags) <: Number ? (time+=lags) : (time+=lags[ev.cmt])
    events[i] = Event(ev.amt,time,ev.evid, ev.cmt, _rate, _duration,
                      ev.ss, ev.ii, ev.base_time, ev.off_event)
  end
  sort!(events)
end

function sorted_approx_unique(event)
  tType = typeof(first(event).time)
  out = Vector{typeof(first(event).time)}(1)
  out[1] = event[1].time
  for i in 2:length(event)
    if abs(out[end] - event[i].time) > 10eps(tType)
      push!(out,event[i].time)
    end
  end
  out
end

function set_value(A :: SVector{L,T}, x,k) where {T,L}
    SVector(ntuple(i->ifelse(i == k, x, A[i]), Val{L}))
end

function increment_value(A :: SVector{L,T}, x,k) where {T,L}
    SVector(ntuple(i->ifelse(i == k, A[i]+x, A[i]), Val{L}))
end

function increment_value(A::Number,x,k)
  A+x
end

function get_magic_args(p,u0,t0)
  if haskey(p,:lags)
    lags = p.lags
  else
    lags = zero(eltype(t0))
  end

  if haskey(p,:bioav)
    bioav = p.bioav
  else
    bioav = 1.0#one(eltype(u0))
  end

  if haskey(p,:rate)
    rate = p.rate
  else
    rate = zero(eltype(u0))
  end

  if haskey(p,:duration)
    duration = p.duration
  else
    duration = zero(eltype(u0))
  end

  lags,bioav,rate,duration
end
