struct SimulatedObservations{S,T,T2}
  subject::S
  times::T
  observed::T2
end

# indexing
@inline function Base.getindex(obs::SimulatedObservations, I...)
  return obs.observed[I...]
end
@inline function Base.setindex!(obs::SimulatedObservations, x, I...)
  obs.observed[I...] = x
end
function DataFrames.DataFrame(obs::SimulatedObservations;
  include_events=true)
  nrows = length(obs.times)
  if include_events
    # Generate the expanded event list
    events = obs.subject.events
    amt  = zeros(typeof(events[1].amt),  nrows)
    evid = zeros(typeof(events[1].evid), nrows)
    cmt  = zeros(typeof(events[1].cmt),  nrows)
    rate = zeros(typeof(events[1].rate), nrows)
    for ev in events
      ind = searchsortedlast(obs.times, ev.time)
      if obs.times[ind] == ev.time
        amt[ind]  = ev.amt
        evid[ind] = ev.evid
        cmt[ind]  = ev.cmt
        rate[ind] = ev.rate
      else
        @warn "Non-observing event encountered in siumulated observation"
      end
    end
    DataFrame(merge(
      (time=obs.times,), obs.observed,
      (amt=amt, evid=evid, cmt=cmt, rate=rate)
    ))
  else
    DataFrame(merge((time=obs.times,), obs.observed))
  end
end

@recipe function f(obs::SimulatedObservations)
  t = obs.times
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.observed)
    push!(names,n)
    push!(plot_vars,v)
  end
  xlabel --> "time"
  legend --> false
  lw --> 3
  ylabel --> reshape(names,1,length(names))
  layout --> good_layout(length(names))
  title --> "Subject ID: $(obs.subject.id)"
  t,plot_vars
end

function good_layout(n)
  n == 1 && return (1,1)
  n == 2 && return (2,1)
  n == 3 && return (3,1)
  n == 4 && return (2,2)
  n > 4 && return (n÷2,2)
end

struct SimulatedPopulation{S}
  sims::S
end
@inline function Base.getindex(pop::SimulatedPopulation, I...)
  return pop.sims[I...]
end
@inline function Base.setindex!(pop::SimulatedPopulation, x, I...)
  pop.sims[I...] = x
end
function DataFrames.DataFrame(pop::SimulatedPopulation)
  dfs = []
  for s in pop.sims
    id = [s.subject.id for i in 1:length(s.times)]
    df = DataFrame(s)
    insertcols!(df, 1, id=id)
    push!(dfs, df)
  end
  reduce(vcat,dfs)
end

@recipe function f(pop::SimulatedPopulation)
  for p in pop.sims
    @series begin
      lw --> 1.5
      title --> "Population Simulation"
      p
    end
  end
  nothing
end
