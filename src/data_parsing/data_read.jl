## Types

struct Person{T1,T2,T3,T4,T5}
  id::Int
  obs::T1
  obs_times::T2
  z::T3
  event_times::T4
  events::T5
end

struct Population{T} <: AbstractVector{T}
  patients::T
end

Base.start(A::Population) = start(A.patients)
Base.next(A::Population, i) = next(A.patients, i)
Base.done(A::Population, i) = done(A.patients, i)

# size
Base.length(A::Population) = length(A.patients)
Base.size(A::Population) = size(A.patients)

# indexing
@inline function Base.getindex(A::Population, I...)
    @boundscheck checkbounds(A.patients, I...)
    @inbounds return A.patients[I...]
end
@inline function Base.setindex!(A::Population, x, I...)
    @boundscheck checkbounds(A.patients, I...)
    @inbounds A.patients[I...] = x
end
Base.indices(A::Population) = indices(A.patients)
Base.IndexStyle(::Type{<:Population}) = Base.IndexLinear()

## Parsing Functions

function generate_person(id,covariates,dvs,raw_data)
  id_start = findfirst(x -> x==id, raw_data[:id])
  id_end   = findlast(x ->  x==id, raw_data[:id])
  person_data = raw_data[id_start:id_end, :]
  z = Dict([sym => person_data[1,sym] for sym in covariates])
  obs_idxs = find(x ->  x==0, person_data[:mdv])
  obs = person_data[obs_idxs,dvs]
  obs_times = person_data[obs_idxs,:time]
  event_idxs = find(x ->  x!=0, person_data[:evid])
  tType = float(eltype(person_data[first(event_idxs),:time]))
  if !haskey(person_data,:addl)
    haskey(person_data,:cmt) ? cmt = person_data[event_idxs,:cmt] : cmt = 1
    event_times = TimeCompartment.(float.(person_data[event_idxs,:time]),cmt,zero(tType))
    evid = person_data[event_idxs,:evid]
    amt = float.(person_data[event_idxs,:amt])
    haskey(person_data,:rate) ? rate = person_data[event_idxs,:rate] : rate = zero(amt)
    haskey(person_data,:ss) ? ss = person_data[event_idxs,:ss] : ss = 0
    events = Event.(amt,evid,cmt,rate,ss)
  else
    # Type is determined by `:amt`, it's unnecessary and can be made Float64
    # If the right conversions are added
    events = Event{typeof(float(person_data[first(event_idxs),:amt]))}[]
    event_times = TimeCompartment{tType}[]
    for i in event_idxs
      addl = person_data[i,:addl]
      curtime = person_data[i,:time]
      for j in 0:addl # 0 means just once
        haskey(person_data,:cmt) ? cmt = person_data[i,:cmt] : cmt = 1
        push!(event_times,TimeCompartment(float(curtime),cmt,zero(float(curtime))))
        amt = float(person_data[i,:amt])
        evid = person_data[i,:evid]
        haskey(person_data,:rate) ? rate = float(person_data[i,:rate]) : rate = zero(amt)
        haskey(person_data,:ss) ? ss = person_data[i,:ss] : ss = 0
        push!(events,Event(amt,evid,cmt,rate,ss))
        if rate != 0
          # Add a turn off event
          duration = amt/rate
          rate_off = curtime + duration
          push!(events,Event(-amt,-1,cmt,-rate,ss))
          push!(event_times,TimeCompartment(rate_off,cmt,duration))
        end
        curtime += person_data[i,:ii]
      end
    end
  end
  order = sortperm(event_times)
  permute!(event_times,order)
  permute!(events,order)
  Person(id,obs,obs_times,z,event_times,events)
end

function process_data(path,covariates,dvs;kwargs...)
  raw_data = readtable(path;kwargs...)
  ids = unique(raw_data[:id])
  pop = Population(map(id->generate_person(id,covariates,dvs,raw_data),ids))
end
