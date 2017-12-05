## Types

struct Person{T1,T2,T3,T4}
  id::Int
  obs::T1
  obs_times::T2
  z::T3
  events::T4
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



"""
    process_data(filename, covariates=Symbol[], dvs=[:dv];
        header=true, names=nothing, separator=nothing)

Import NONMEM-formatted data from `filename`.

- `covariates` and `dvs` are the list of columns which are considered covariates and dependent variables.
- `header`: whether or not there is a header row (default=`true`)
- `names`: a list names of colums. If `nothing` (default) then these are taken as the first row (`header` must be `true`)
- `separator`: the delimiter. Should be either a character (e.g. `','`) or `nothing` (default) in which case it is taken to be whitespace-delimited with repeated whitespaces treated as one delimiter.

"""
function process_data(filename,covariates=Symbol[],dvs=Symbol[:dv];
    header=true, names=nothing, separator=nothing)
    io = open(filename)
    if header == true
        l = readline(io)
        if names === nothing
            if separator === nothing
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
    if separator === nothing
        str_mat = readdlm(io,String)
    else
        str_mat = readdlm(io,separator,String)
    end
    close(io)

    @assert length(names) == size(str_mat,2)
    cols = Dict(name => str_mat[:,i] for (i,name) in enumerate(names))
    build_dataset(cols,covariates,dvs)
end

function build_dataset(;kwargs...)
    if typeof(kwargs[1][2]) <: Number
        _kwargs = [(kw[1],[string.(kw[2])]) for kw in kwargs]
    else
        _kwargs = [(kw[1],string.(kw[2])) for kw in kwargs]
    end
    cols = Dict(_kwargs)
    m = length(_kwargs[1][2])
    for k in keys(cols)
        @assert length(cols[k]) == m
    end

    if !haskey(cols,:id)
        cols[:id] = ["1" for i in 1:m]
    end
    if !haskey(cols,:time)
        cols[:time] = ["0.0" for i in 1:m]
    end
    if !haskey(cols,:evid)
        cols[:evid] = ["1" for i in 1:m]
    end
    data = build_dataset(cols)
    if length(data.patients)==1
        return data[1]
    else
        return data
    end
end

function build_dataset(cols,covariates=(),dvs=())

    names = collect(keys(cols))
    m = length(cols[:id])

    if covariates isa Vector{Int}
        covariates = names[covariates]
    end
    if dvs isa Vector{Int}
        dvs = names[dvs]
    end

    ## Common fields
    eltype(cols[:id]) <: AbstractString ? ids = parse.(Int, cols[:id]) : ids = cols[:id]
    eltype(cols[:time]) <: AbstractString ? times = parse.(Float64, cols[:time]) : times = cols[:time]
    eltype(cols[:evid]) <: AbstractString ? evids = parse.(Int8, cols[:evid]) : evids = cols[:evid]

    uids  = unique(ids)
    !isempty(dvs) && (Tdv = NamedTuples.create_namedtuple_type(dvs))
    !isempty(covariates) && (Tcv = NamedTuples.create_namedtuple_type(covariates))

    subjects = map(uids) do id
        ## Observations
        idx_obs = filter(i -> ids[i] == id && evids[i] == 0, 1:m)
        obs_times = times[idx_obs]
        isempty(dvs) ? obs = nothing : obs = Tdv.([misparse.(Float64, cols[dv][idx_obs]) for dv in dvs]...)

        ## Covariates
        #  TODO: allow non-constant covariates
        i_z = findfirst(x -> x ==id, ids)
        isempty(covariates) ? z = nothing : z = Tcv([parse(Float64, cols[cv][i_z]) for cv in covariates]...)

        ## Events
        idx_evt = filter(i -> ids[i] ==id && evids[i] != 0, 1:m)
        events = Event{Float64}[]

        for i in idx_evt
            t    = times[i]
            evid = evids[i]
            amt  = misparse(Float64, cols[:amt][i]) # can be missing if evid=2
            addl = haskey(cols, :addl) ? parse(Int, cols[:addl][i]) : 0
            ii   = haskey(cols, :ii)   ? parse(Float64, cols[:ii][i]) : 0.0
            cmt  = haskey(cols,:cmt)   ? parse(Int, cols[:cmt][i])  : 1
            rate = haskey(cols,:rate)  ? misparse(Float64, cols[:rate][i]) : 0.0
            ss   = haskey(cols,:ss)    ? misparse(Int8, cols[:ss][i])   : Int8(0)

            for j = 0:addl  # addl==0 means just once
                duration = amt/rate
                @assert amt != 0 || ss == 1 || evid == 2
                if amt == 0 && evid != 2
                    @assert rate > 0
                    # These are dose events having AMT=0, RATE>0, SS=1, and II=0.
                    # Such an event consists of infusion with the stated rate,
                    # starting at time −∞, and ending at the time on the dose
                    # ev event record. Bioavailability fractions do not apply
                    # to these doses.

                    # Put in a fake ii=10.0 for the steady state interval length
                    ii == 0.0 && (ii = 10.0)
                    push!(events,Event(amt,0.0,evid,cmt,rate,ii,ss,ii,t,Int8(1)))
                    if rate != 0 && ss == 0
                        push!(events,Event(amt,t,Int8(-1),cmt,rate,duration,Int8(0),ii,t,Int8(-1)))
                    end
                else
                    push!(events,Event(amt,t,evid,cmt,rate,duration,ss,ii,t,Int8(1)))
                    if rate != 0 && ss == 0
                        push!(events,Event(amt,t + duration,Int8(-1),cmt,rate,duration,ss,ii,t,Int8(-1)))
                    end
                end
                t += ii
            end
        end

        sort!(events)
        Person(id, obs, obs_times, z, events)
    end
    Population(subjects)
end
