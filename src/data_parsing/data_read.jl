Base.iterate(A::Population)    = iterate(A.subjects)
Base.iterate(A::Population, i) = iterate(A.subjects, i)

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
Base.axes(A::Population) = axes(A.subjects)
Base.IndexStyle(::Type{<:Population}) = Base.IndexLinear()

"""
    process_nmtran(data, cvs=[], dvs=[:dv])

Import NMTRAN-formatted data.

- `cvs` covariates specified by either names or column numbers
- `dvs` dependent variables specified by either names or column numbers

"""
function process_nmtran(data,cvs=Symbol[],dvs=Symbol[:dv])
    Names = names(data)
    m = size(data, 1)

    if :id ∉ Names
        data[:id] = 1
    end
    if :time ∉ Names
        data[:time] = 0.0
    end
    if :evid ∉ Names
        data[:evid] = Int8(1)
    end

    if cvs isa AbstractVector{<:Integer}
        cvs = names[cvs]
    end
    if dvs isa AbstractVector{<:Integer}
        dvs = names[dvs]
    end

    ## Common fields

    uids  = unique(data[:id])
    Tdv = isempty(dvs) ? Nothing : Core.NamedTuple{(dvs...,),NTuple{length(dvs),Float64}}
    Tcv = isempty(cvs) ? Nothing : Core.NamedTuple{(cvs...,),NTuple{length(cvs),Float64}}

    subjects = map(uids) do id
        ## Observations
        idx_obs = filter(i -> data[:id][i] == id && data[:evid][i] == 0, 1:m)
        obs_times = data[:time][idx_obs]

        dv_idx = [ data[dv][idx_obs] for dv in dvs]

        obs_dvs = Tdv === Nothing ? nothing : map((x...) -> Tdv(x), dv_idx...)
        obs_cmts = :cmt ∈ Names ? data[:cmt][idx_obs] : nothing
        observations = Observation.(obs_times, obs_dvs, obs_cmts)

        ## Covariates
        #  TODO: allow non-constant covariates
        i_z = findfirst(x -> x == id, data[:id])
        covariates = Tcv === Nothing ? nothing : Tcv( data[cv][i_z] for cv in cvs)

        ## Events
        idx_evt = filter(i -> data[:id][i] == id && data[:evid][i] != 0, 1:m)
        events = Event[]

        for i in idx_evt
            t    = float(data[:time][i])
            evid = Int8(data[:evid][i])
            amt  = :amt  ∈ Names ? float(data[:amt][i])  : nothing # can be missing if evid=2
            addl = :addl ∈ Names ? Int(data[:addl][i])   : 0
            ii   = :ii   ∈ Names ? float(data[:ii][i])   : 0.0
            cmt  = :cmt  ∈ Names ? Int(data[:cmt][i])    : 1
            rate = :rate ∈ Names ? float(data[:rate][i]) : 0.0
            ss   = :ss   ∈ Names ? Int8(data[:ss][i])    : Int8(0)

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

                    push!(events,Event(amt, t, evid, cmt, rate, ii, _ss, ii, t, Int8(1)))
                else
                    push!(events,Event(amt, t, evid, cmt, rate, duration, _ss, ii, t, Int8(1)))
                    if rate != 0 && _ss == 0
                        push!(events,Event(amt, t + duration, Int8(-1), cmt, rate, duration, _ss, ii, t, Int8(-1)))
                    end
                end
                t += ii
            end
        end

        sort!(events)
        Subject(id = id, obs = observations, cvs = covariates, evs = events)
    end
    Population(subjects)
end
build_observation_list(obs::AbstractVector{<:Observation}) = obs
function build_observation_list(obs::AbstractDataFrame)
    isempty(obs) && return Observation[]
    cmt = :cmt ∈ names(obs) ? obs[:cmt] : 1
    time = obs[:time]
    vars = setdiff(names(obs), (:time, :cmt))
    if isa(time, Unitful.Time)
        time = convert.(Float64, getfield(uconvert.(u"hr", time), :val))
    else
        time = float(time)
    end
    return Observation.(time,
                        NamedTuple{Tuple(vars),NTuple{length(vars),Float64}}.(values.(eachrow(obs[vars]))),
                        cmt)
end

build_event_list(evs::AbstractVector{<:Event}) = evs
function build_event_list(regimen::DosageRegimen)
    data = getfield(regimen, :data)
    events = Event[]
    for i in 1:size(data, 1)
        t    = data[:time][i]
        evid = data[:evid][i]
        amt  = data[:amt][i]
        addl = data[:addl][i]
        ii   = data[:ii][i]
        cmt  = data[:cmt][i]
        rate = data[:rate][i]
        ss   = data[:ss][i]

        for j = 0:addl  # addl==0 means just once
            _ss = iszero(j) ? ss : zero(Int8)
            duration = amt/rate
            @assert amt != 0 || _ss == 1 || evid == 2
            if iszero(amt) && evid != 2
                @assert rate > 0
                # These are dose events having AMT=0, RATE>0, SS=1, and II=0.
                # Such an event consists of infusion with the stated rate,
                # starting at time −∞, and ending at the time on the dose
                # ev event record. Bioavailability fractions do not apply
                # to these doses.
                push!(events, Event(amt, t, evid, cmt, rate, ii, _ss, ii, t, Int8(1)))
            else
                push!(events, Event(amt, t, evid, cmt, rate, duration, _ss, ii, t, Int8(1)))
                if rate != 0 && _ss == 0
                    push!(events, Event(amt, t + duration, Int8(-1), cmt, rate, duration, _ss, ii, t, Int8(-1)))
                end
            end
            t += ii
        end
    end
    sort!(events)
end
