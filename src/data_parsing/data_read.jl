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
misparse(T,x) = x == "." ? zero(T) : parse(T,x)

using NamedTuples

function process_data(filename,covariates=Symbol[],dvs=Symbol[:dv]; header=true, names=nothing, separator="")
    io = open(filename)
    if header == true
        l = readline(io)
        if names == nothing
            if separator == ""
                names = lowercase.(split(l))
            else
                names = lowercase.(split(l,separator))
            end
            if names[1] == "#"
                shift!(names)
            end
            names = Symbol.(names)
        end
    end
    if separator == ""
        str_mat = readdlm(io,String)
    else
        str_mat = readdlm(io,separator,String)
    end
    close(io)

    m, n = size(str_mat)

    @assert length(names) == size(str_mat,2)
    cols = Dict(name => str_mat[:,i] for (i,name) in enumerate(names))

    if covariates isa Vector{Int}
        covariates = names[covariates]
    end
    if dvs isa Vector{Int}
        dvs = names[dvs]
    end

    ## Common fields
    ids   = parse.(Int, cols[:id])
    uids  = unique(ids)

    times = parse.(Float64, cols[:time])
    evids = parse.(Int,  cols[:evid])

    Tdv = NamedTuples.create_namedtuple_type(dvs)
    Tcv = NamedTuples.create_namedtuple_type(covariates)

    subjects = map(uids) do id
        ## Observations
        idx_obs = filter(i -> ids[i] == id && evids[i] == 0, 1:m)
        obs_times = times[idx_obs]
        obs = Tdv.([misparse.(Float64, cols[dv][idx_obs]) for dv in dvs]...)

        ## Covariates
        #  TODO: allow non-constant covariates
        i_z = findfirst(x -> x ==id, ids)
        z = Tcv([parse(Float64, cols[cv][i_z]) for cv in covariates]...)

        ## Events
        idx_evt = filter(i -> ids[i] ==id && evids[i] != 0, 1:m)
        if !haskey(cols, :addl)
            # each row corresponds to 1 event
            amt  = misparse.(Float64, cols[:amt][idx_evt]) # can be missing if evid=2
            cmt  = haskey(cols,:cmt)  ? parse.(Int, cols[:cmt][idx_evt]) : 1
            rate = haskey(cols,:rate) ? misparse.(Float64, cols[:cmt][idx_evt]) : 0.0
            ss   = haskey(cols,:ss)   ? misparse.(Int, cols[:ss][idx_evt])  : 0

            events = Event.(amt, evids[idx_evt], cmt, rate, ss)
            event_times = TimeCompartment.(times[idx_evt], cmt, 0.0)
        else
            # allow for repeated events
            event_times = TimeCompartment{Float64}[]
            events = Event{Float64}[]

            for i in idx_evt
                t    = times[i]
                evid = evids[i]
                addl = parse(Int, cols[:addl][i])
                amt  = parse(Float64, cols[:amt][i])
                ii   = parse(Float64, cols[:ii][i])
                cmt  = haskey(cols,:cmt)  ? parse(Int, cols[:cmt][i])  : 1
                rate = haskey(cols,:rate) ? parse(Float64, cols[:rate][i]) : 0.0
                ss   = haskey(cols,:ss)   ? parse(Int, cols[:ss][i])   : 0

                for j = 0:addl  # addl==0 means just once
                    push!(event_times, TimeCompartment(t,cmt,0.0))
                    push!(events,      Event(amt,evid,cmt,rate,ss))
                    if rate != 0
                        duration = amt/rate
                        push!(event_times, TimeCompartment(t + duration,cmt,duration))
                        push!(events,      Event(-amt,-1,cmt,-rate,ss))
                    end

                    t += ii
                end
            end
        end

        order = sortperm(event_times)
        permute!(event_times,order)
        permute!(events,order)
        Person(id, obs, obs_times, z, event_times, events)
    end
    Population(subjects)
end
