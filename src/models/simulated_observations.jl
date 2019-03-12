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
function DataFrames.DataFrame(obs::SimulatedObservations)
  DataFrame(merge((time=obs.times,),obs.observed))
end

@recipe function f(obs::SimulatedObservations)
  t = obs.times
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.obs)
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
  dfs = [DataFrame(merge((id=[s.subject.id for i in 1:length(s.times)],
                          time=s.times),
                          s.observed)) for s in pop.sims]
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
