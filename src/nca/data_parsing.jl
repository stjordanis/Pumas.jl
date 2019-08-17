using CSV, DataFrames

"""
    read_nca(df::Union{DataFrame,AbstractString}; id=:id, time=:time,
             amt=:amt, route=:route, duration=:duration, blq=:blq,
             group=nothing, ii=nothing, concu=true, timeu=true, amtu=true, verbose=true, kwargs...)

Parse a `DataFrame` object or a CSV file to `NCAPopulation`. `NCAPopulation`
holds an array of `NCASubject`s which can cache certain results to achieve
efficient NCA calculation.
"""
read_nca(file::AbstractString; kwargs...) = read_nca(CSV.read(file); kwargs...)
# TODO: add ploting time
# TODO: infusion
# TODO: plot time
function read_nca(df; group=nothing, kwargs...)
  pop = if group === nothing
    ___read_nca(df; kwargs...)
  else
    dfs = groupby(df, group)
    groupnum = length(dfs)
    dfpops = map(dfs) do df
      if group isa AbstractArray && length(group) > 1
        grouplabel = map(string, group)
        groupnames = map(string, first(df[!,group]))
        currentgroup = map(=>, grouplabel, groupnames)
      else
        group isa Symbol || ( group = first(group) )
        grouplabel = string(group)
        groupnames = first(df[!,group])
        currentgroup = grouplabel => groupnames
      end
      pop = ___read_nca(df; group=currentgroup, kwargs...)
      return pop
    end
    pops = map(i->dfpops[i][!, end], 1:groupnum)
    NCAPopulation(vcat(pops...))
  end
  return pop
end

function ___read_nca(df; id=:id, time=:time, conc=:conc, occasion=:occasion,
                     start_time=:start_time, end_time=:end_time, volume=:volume,
                     amt=:amt, route=:route,#= rate=nothing,=# duration=:duration, blq=:blq,
                     ii=:ii, ss=:ss, group=nothing, concu=true, timeu=true, amtu=true, volumeu=true,
                     verbose=true, kwargs...)
  local ids, times, concs, amts
  dfnames = names(df)
  has_id = id in dfnames
  has_time = time in dfnames
  has_conc = conc in dfnames
  if has_id && has_time && has_conc
    urine = false
  else
    has_start_time = start_time in dfnames
    has_end_time   = end_time   in dfnames
    has_volume     = volume     in dfnames
    if has_start_time && has_end_time && has_volume && has_conc
      urine = true
    else
      @info "The CSV file has keys: $(names(df))"
      throw(ArgumentError("The CSV file must have: `id, time, conc` or `id, start_time, end_time, volume, conc` columns"))
    end
  end
  blq = (blq in dfnames) ? blq : nothing
  amt = (amt in dfnames) ? amt : nothing
  ii  = (ii in dfnames) ? ii : nothing
  ss  = (ss in dfnames) ? ss : nothing
  route = (route in dfnames) ? route : nothing
  occasion = (occasion in dfnames) ? occasion : nothing
  duration = (duration in dfnames) ? duration : nothing
  hasdose = amt !== nothing && route !== nothing
  if verbose
    hasdose || @warn "No dosage information has passed. If the dataset has dosage information, you can pass the column names by `amt=:AMT, route=:route`."
  end

  # BLQ
  @noinline blqerr() = throw(ArgumentError("blq can only be 0 or 1"))
  if blq !== nothing
    blqs = df[!,blq]
    eltype(blqs) <: Union{Int, Bool} || blqerr()
    extrema(blqs) == (0, 1) || blqerr()
    df = deleterows!(deepcopy(df), findall(isequal(1), blqs))
  end

  sortvars = urine ? (occasion === nothing ? (id, start_time, end_time) : (id, start_time, end_time, occasion)) :
                     (occasion === nothing ? (id, time) : (id, time, occasion))
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sort(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  ids   = df[!,id]
  if urine
    start_time′ = df[!,start_time]
    end_time′ = df[!,end_time]
    Δt = @. end_time′ - start_time′
    times = @. start_time′ + Δt
  else
    start_time′ = end_time′ = Δt = nothing
    times = df[!,time]
  end
  concs = df[!,conc]
  amts  = amt === nothing ? nothing : df[!,amt]
  iis  = ii === nothing ? nothing : df[!,ii]
  sss  = ss === nothing ? nothing : df[!,ss]
  occasions = occasion === nothing ? nothing : df[!,occasion]
  uids = unique(ids)
  idx  = -1
  # FIXME! This is better written as map(uids) do id it currently triggers a dispatch bug in Julia via CSV
  ncas = Vector{Any}(undef, length(uids))
  for (i, id) in enumerate(uids)
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    # the time array for the i-th subject
    subjtime = @view(times[idx])
    if hasdose
      dose_idx = findall(x->x !== missing && x > zero(x), @view amts[idx])
      length(dose_idx) > 1 && occasion === nothing && error("`occasion` must be provided for multiple dosing data")
      # We want to use time instead of an integer index here, because later we
      # need to remove BLQ and missing data, so that an index number will no
      # longer be valid.
      if length(dose_idx) == 1
        dose_idx = dose_idx[1]
        dose_time = subjtime[dose_idx[1]]
      else
        dose_time = similar(times, Base.nonmissingtype(eltype(times)), length(dose_idx))
        for (n,i) in enumerate(dose_idx)
          dose_time[n] = subjtime[i]
        end
      end
      route′ = map(dose_idx) do i
        routei = lowercase(df[!,route][i])
        routei == "iv" ? IVBolus :
          routei == "inf" ? IVInfusion :
          routei == "ev" ? EV :
          throw(ArgumentError("route can only be `iv`, `ev`, or `inf`"))
      end
      ii = map(i -> iis === nothing ? false : iis[i], dose_idx)
      ss = map(dose_idx) do i
        sss === nothing ? false :
          sss[i] == 0 ? false :
          sss[i] == 1 ? true :
          throw(ArgumentError("ss can only be 0 or 1"))
      end
      duration′ = duration === nothing ? nothing : df[!,duration][dose_idx]*timeu
      doses = NCADose.(dose_time*timeu, amts[dose_idx]*amtu, duration′, route′, ii*timeu, ss)
    #elseif occasion !== nothing
    #  subjoccasion = @view occasions[idx]
    #  occs = unique(subjoccasion)
    #  doses = map(occs) do occ
    #    dose_idx = findfirst(isequal(occ), subjoccasion)
    #    dose_time = subjtime[dose_idx]
    #    ii = iis === nothing ? false : iis[i]
    #    NCADose(dose_time*timeu, zero(amtu), ii*timeu, DosingUnknown)
    #  end
    else
      doses = nothing
    end
    try
      ncas[i] = NCASubject(concs[idx], times[idx]; id=id, group=group, dose=doses, concu=concu, timeu=timeu, volumeu=volumeu,
                           start_time=start_time′, end_time=end_time′,
                           volume=urine ? df[!,volume][idx] : nothing,
                           concblq=blq===nothing ? nothing : :keep, kwargs...)
    catch
      @info "ID $id errored"
      rethrow()
    end
  end
  # Use broadcast to tighten ncas element type
  pop = NCAPopulation(identity.(ncas))
  return pop
end
