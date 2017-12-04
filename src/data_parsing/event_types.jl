struct Event{T,T2}
  amt::T
  time::T2
  evid::Int
  cmt::Int
  rate::T
  ss::Int
  ii::T2
  base_time::T2 # So that this is kept after modifications to duration and rate
  off_event::Bool
end

Base.isless(a::Event,b::Event) = isless(a.time,b.time)
Base.isless(a::Event,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::Event) = isless(a,b.time)

function change_times!(events,lags,bioav,rate,duration)
  @show lags
  for i in eachindex(events)
    ev = events[i]
    ev.rate != 0 ? duration = ev.amt/ev.rate : duration = zero(ev.amt)
    if ev.off_event
      typeof(bioav) <: Number ? time = ev.base_time + bioav*duration :
                                time = ev.base_time + bioav[ev.cmt]*duration
    else
      time = ev.base_time
    end
    typeof(lags) <: Number ? (time+=lags) : (time+=lags[ev.cmt])
    events[i] = Event(ev.amt,time,ev.evid, ev.cmt, ev.rate,
                      ev.ss, ev.ii, ev.base_time, ev.off_event)
  end
  events
end

function Base.vcat(a::Number,event::AbstractArray{<:Event})
  if a == event[1].time
    out = Vector{typeof(a)}(length(event))
    for i in 1:length(event)
      out[i] = event[i].time
    end
  else
    out = Vector{typeof(a)}(length(event)+1)
    out[1] = a
    for i in 1:length(event)
      out[i+1] = event[i].time
    end
  end
  out
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
