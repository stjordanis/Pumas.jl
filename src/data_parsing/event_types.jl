struct Event{T,T2}
  amt::T
  evid::Int
  cmt::Int
  rate::T
  ss::Int
  ii::T2
end

struct TimeCompartment{T}
  time::T
  compartment::Int
  duration::T
  rate::T
end

Base.isless(a::TimeCompartment,b::TimeCompartment) = isless(a.time,b.time)
Base.isless(a::TimeCompartment,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::TimeCompartment) = isless(a,b.time)
Base.:+(a::TimeCompartment,b::Number) = TimeCompartment(a.time+b,a.compartment,a.duration,a.rate)
function remove_lags(events,event_times,lags::Number,bioav,rate,duration)
  new_event_times = event_times +  lags
  change_duration_by_bioav!(new_event_times,bioav,rate,duration)
  if bioav != 1
    order = sortperm(new_event_times)
    permute!(new_event_times,order)
    permute!(events,order)
  end
  new_event_times,events
end
function remove_lags(events,event_times,lags::AbstractArray,bioav,rate,duration)
  new_event_times = Vector{eltype(event_times)}(length(event_times))
  for i in event_times
    tci = event_times[i]
    new_event_times[i] = tci.time + lags[tci.compartment]
  end
  change_duration_by_bioav!(new_event_times,bioav,rate,duration)
  order = sortperm(new_event_times)
  permute!(new_event_times,order)
  permute!(events,order)
  new_event_times,permute(events,order)
end

function change_duration_by_bioav!(event_times,bioav,rate,duration)
  if bioav != 1
    for i in eachindex(event_times)
      ev = event_times[i]
      if typeof(bioav) <: Number
        event_times[i] = TimeCompartment(ev.time - (1-bioav)*ev.duration,
                                         ev.compartment,ev.duration,ev.rate)
      else
        event_times[i] = TimeCompartment(ev.time - (1-bioav[ev.compartment])*ev.duration,
                                         ev.compartment,ev.duration,ev.rate)
      end
    end
  end
  event_times
end

function Base.vcat(a::Number,event_times::AbstractArray{<:TimeCompartment})
  if a == event_times[1].time
    out = Vector{typeof(a)}(length(event_times))
    for i in 1:length(event_times)
      out[i] = event_times[i].time
    end
  else
    out = Vector{typeof(a)}(length(event_times)+1)
    out[1] = a
    for i in 1:length(event_times)
      out[i+1] = event_times[i].time
    end
  end
  out
end

function sorted_approx_unique(event_times)
  tType = typeof(first(event_times).time)
  out = Vector{typeof(first(event_times).time)}(1)
  out[1] = event_times[1].time
  for i in 2:length(event_times)
    if abs(out[end] - event_times[i].time) > 10eps(tType)
      push!(out,event_times[i].time)
    end
  end
  out
end
