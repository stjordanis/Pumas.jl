## Types

struct Subject{T1,T2,T3}
  id::Int
  observations::T1
  covariates::T2
  events::T3
end

struct Population{T} <: AbstractVector{T}
  subjects::T
end

struct Observation{T,V,C}
    time::T
    val::V
    cmt::C
end


function timespan(sub::Subject)
    lo, hi = extrema(evt.time for evt in sub.events)
    if !isempty(sub.observations)
        obs_lo, obs_hi = extrema(obs.time for obs in sub.observations)
        lo = min(lo, obs_lo)
        hi = max(hi, obs_hi)
    end
    lo, hi+sqrt(eps())
end



Base.start(A::Population) = start(A.subjects)
Base.next(A::Population, i) = next(A.subjects, i)
Base.done(A::Population, i) = done(A.subjects, i)

# size
Base.length(A::Population) = length(A.subjects)
Base.size(A::Population) = size(A.subjects)

# indexing
@inline function Base.getindex(A::Population, I...)
    @boundscheck checkbounds(A.subjects, I...)
    @inbounds return A.subjects[I...]
end
@inline function Base.setindex!(A::Population, x, I...)
    @boundscheck checkbounds(A.subjects, I...)
    @inbounds A.subjects[I...] = x
end
Base.indices(A::Population) = indices(A.subjects)
Base.IndexStyle(::Type{<:Population}) = Base.IndexLinear()

## Parsing Functions
misparse(T,x) = x == "." ? zero(T) : parse(T,x)



"""
    process_data(filename, cvs=[], dvs=[:dv];
        header=true, names=nothing, separator=nothing)

Import NONMEM-formatted data from `filename`.

- `cvs` and `dvs` are the list of columns (either names or column numbers) which are considered covariates and dependent variables.
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
    if length(data.subjects)==1
        return data[1]
    else
        return data
    end
end

function build_dataset(cols,cvs=[],dvs=[:dv])

    names = collect(keys(cols))
    m = length(cols[:id])

    if cvs isa Vector{Int}
        cvs = names[cvs]
    end
    if dvs isa Vector{Int}
        dvs = names[dvs]
    end

    ## Common fields
    ids   = eltype(cols[:id]) <: AbstractString ?   parse.(Int, cols[:id])       : cols[:id]
    times = eltype(cols[:time]) <: AbstractString ? parse.(Float64, cols[:time]) : cols[:time]
    evids = eltype(cols[:evid]) <: AbstractString ? parse.(Int8, cols[:evid])    : cols[:evid]

    uids  = unique(ids)
    Tdv = isempty(dvs) ? Void : NamedTuples.create_namedtuple_type(dvs)
    Tcv = isempty(cvs) ? Void : NamedTuples.create_namedtuple_type(cvs)

    subjects = map(uids) do id
        ## Observations
        idx_obs = filter(i -> ids[i] == id && evids[i] == 0, 1:m)
        obs_times = times[idx_obs]

        obs_dvs = Tdv.([parse.(Float64, cols[dv][idx_obs]) for dv in dvs]...)
        obs_cmts = haskey(cols,:cmt) ? misparse.(Int, cols[:cmt][idx_obs]) : nothing
        observations = Observation.(obs_times, obs_dvs, obs_cmts)

        ## Covariates
        #  TODO: allow non-constant covariates
        i_z = findfirst(x -> x ==id, ids)
        covariates = Tcv([parse(Float64, cols[cv][i_z]) for cv in cvs]...)

        ## Events
        idx_evt = filter(i -> ids[i] ==id && evids[i] != 0, 1:m)
        events = Event{Float64,Float64,Float64}[]

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
                j == 0 ? _ss = ss : _ss = Int8(0)
                duration = amt/rate
                @assert amt != 0 || _ss == 1 || evid == 2
                if amt == 0 && evid != 2
                    @assert rate > 0
                    # These are dose events having AMT=0, RATE>0, SS=1, and II=0.
                    # Such an event consists of infusion with the stated rate,
                    # starting at time −∞, and ending at the time on the dose
                    # ev event record. Bioavailability fractions do not apply
                    # to these doses.

                    # Put in a fake ii=10.0 for the steady state interval length
                    ii == 0.0 && (ii = 10.0)
                    push!(events,Event(amt,t,evid,cmt,rate,ii,_ss,ii,t,Int8(1)))
                else
                    push!(events,Event(amt,t,evid,cmt,rate,duration,_ss,ii,t,Int8(1)))
                    if rate != 0 && _ss == 0
                        push!(events,Event(amt,t + duration,Int8(-1),cmt,rate,duration,_ss,ii,t,Int8(-1)))
                    end
                end
                t += ii
            end
        end

        sort!(events)
        Subject(id, observations, covariates, events)
    end
    Population(subjects)
end
