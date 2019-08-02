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

# DataFrame conversion
function DataFrames.DataFrame(obs::SimulatedObservations;
  include_events=true, event_order_reverse=true,
  include_covariates=true)
  nrows = length(obs.times)
  events = obs.subject.events
  nev = events isa Array ? length(events) : 0
  evtimes = map(ev->ev.time, events)
  times = obs.times
  ntime = length(obs.times)
  observed = obs.observed
  df = DataFrame(time=deepcopy(times))
  for k in keys(observed)
    var = observed[k]
    lenvar = length(var)
    if lenvar != ntime && lenvar == nev # NCA quantities
      # pad NCA quantities
      i = 0
      var = map(times) do t
        i1 = min(i+1, nev)
        t >= evtimes[i1] && (i = i1)
        var[i1]
      end
    end
    df[!,k] = deepcopy(var)
  end
  obs_columns = [keys(obs.observed)...]
  if include_events
    # Append event columns
    ## For observations we have `evid=0` and `cmt=0`, the latter
    ## subject to changes in the future
    df[!,:amt]  = zeros(typeof(events[1].amt),  nrows)
    df[!,:evid] = zeros(typeof(events[1].evid), nrows)
    df[!,:cmt]  = missings(typeof(events[1].cmt), nrows)
    df[!,:rate] = zeros(typeof(events[1].rate), nrows)
    # Add rows corresponding to the events
    ## For events that have a matching observation at the event
    ## time, the values of the derived variables are copied.
    ## Otherwise they are set to `missing`.
    for ev in events
      ind = searchsortedlast(obs.times, ev.time)
      if ind != 0 && obs.times[ind] == ev.time
        ev_row = vcat(ev.time, df[ind, obs_columns]...,
                      ev.amt, ev.evid, ev.cmt, ev.rate)
        push!(df, ev_row)
      else
        ev_row = vcat(ev.time, missings(length(obs_columns)),
                      ev.amt, ev.evid, ev.cmt, ev.rate)
        try
          push!(df, ev_row)
        catch # exception if df doesn't allow missing values yet
          allowmissing!(df, obs_columns)
          push!(df, ev_row)
        end
      end
    end
    sort!(df, (:time, order(:evid, rev=event_order_reverse)))
  end
  if include_covariates
    covariates = obs.subject.covariates
    if covariates != nothing
      for (cov, value) in pairs(covariates)
        df[!,cov] .= value
      end
    end
  end
  df
end

@recipe function f(obs::SimulatedObservations; obsnames=nothing)
  t = obs.times
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.observed)
    if obsnames !== nothing
      !(n in obsnames) && continue
    end
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
function DataFrames.DataFrame(pop::SimulatedPopulation; kwargs...)
  dfs = []
  for s in pop.sims
    df = DataFrame(s; kwargs...)
    id = [s.subject.id for i in 1:size(df, 1)]
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
