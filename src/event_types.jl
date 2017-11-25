struct Event{T}
  amt::T
  evid::Int
  cmt::Int
  rate::T
end

struct TimeCompartment{T}
  time::T
  compartment::Int
end

Base.isless(a::TimeCompartment,b::TimeCompartment) = isless(a.time,b.time)
Base.isless(a::TimeCompartment,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::TimeCompartment) = isless(a,b.time)
Base.:+(a::TimeCompartment,b::Number) = TimeCompartment(a.time+b,a.compartment)
function remove_lags(events,event_times,lags::Number)
  event_times +  lags,events
end
function remove_lags(events,event_times,lags::AbstractArray)
  new_event_times = Vector{eltype(event_times)}(length(event_times))
  for i in event_times
    tci = event_times[i]
    new_event_times[i] = tci.time + lags[tci.compartment]
  end
  order = sortperm(new_event_times)
  permute!(new_event_times,order)
  permute!(events,order)
  new_event_times,permute(events,order)
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
