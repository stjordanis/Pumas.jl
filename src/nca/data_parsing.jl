using CSV, DataFrames

"""
    read_nca(df::Union{DataFrame,AbstractString}; id=:ID, time=:time,
      conc=:conc, occasion=nothing, amt=nothing, route=nothing, duration=nothing,
      ii=nothing, concu=true, timeu=true, amtu=true, warn=true, kwargs...) -> NCAPopulation

Parse a `DataFrame` object or a CSV file to `NCAPopulation`. `NCAPopulation`
holds an array of `NCASubject`s which can cache certain results to achieve
efficient NCA calculation.
"""
read_nca(file::AbstractString; kwargs...) = read_nca(CSV.read(file); kwargs...)
# TODO: add ploting time
# TODO: infusion
# TODO: plot time
function read_nca(df; group=nothing, ii=nothing, kwargs...)
  pop, added_ii = if group === nothing
    ___read_nca(df; ii=ii, kwargs...)
  else
    dfs = groupby(df, group)
    groupnum = length(dfs)
    added_ii = true
    dfpops = map(dfs) do df
      if group isa AbstractArray && length(group) > 1
        grouplabel = map(string, group)
        groupnames = map(string, first(df[group]))
        currentgroup = map(=>, grouplabel, groupnames)
      else
        group isa Symbol || ( group = first(group) )
        grouplabel = string(group)
        groupnames = first(df[group])
        currentgroup = grouplabel => groupnames
      end
      pop, tmp = ___read_nca(df; group=currentgroup, ii=ii, kwargs...)
      added_ii &= tmp
      return pop
    end
    pops = map(i->dfpops[i][end], 1:groupnum)
    NCAPopulation(vcat(pops...)), added_ii
  end
  ii !== nothing && (added_ii || add_ii!(pop, ii)) # avoiding recomputing the fast path
  return pop
end

function ___read_nca(df; id=:ID, group=nothing, time=:time, conc=:conc, occasion=nothing,
                       amt=nothing, route=nothing,# rate=nothing,
                       duration=nothing, ii=nothing, concu=true, timeu=true, amtu=true, warn=true, kwargs...)
  local ids, times, concs, amts
  try
    df[id]
    df[time]
    df[conc]
    amt === nothing ? nothing : df[amt]
    route === nothing ? nothing : df[route]
    #rate === nothing ? nothing : df[rate]
    duration === nothing ? nothing : df[duration]
  catch x
    @info "The CSV file has keys: $(names(df))"
    throw(x)
  end
  hasdose = amt !== nothing && route !== nothing
  if warn
    hasdose || @warn "No dosage information has passed. If the dataset has dosage information, you can pass the column names by `amt=:AMT, route=:route`."
  end
  sortvars = occasion === nothing ? (id, time) : (id, time, occasion)
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sort(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  ids   = df[id]
  times = df[time]
  concs = df[conc]
  amts  = amt === nothing ? nothing : df[amt]
  occasions = occasion === nothing ? nothing : df[occasion]
  uids = unique(ids)
  idx  = -1
  # FIXME! This is better written as map(uids) do id it currently triggers a dispatch bug in Julia via CSV
  ncas = Vector{Any}(undef, length(uids))
  for (i, id) in enumerate(uids)
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    # the time array for the i-th subject
    subjtime = @view times[idx]
    if hasdose
      dose_idx = findall(x->!ismissing(x) && x > zero(x), @view amts[idx])
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
        routei = df[route][i]
        routei == "iv" ? IVBolus :
          routei == "inf" ? IVInfusion :
          routei == "ev" ? EV :
          throw(ArgumentError("route can only be `iv`, `ev`, or `inf`"))
      end
      duration′ = duration === nothing ? nothing : df[duration][dose_idx]*timeu
      doses = NCADose.(dose_time*timeu, amts[dose_idx]*amtu, duration′, route′)
    elseif occasion !== nothing
      subjoccasion = @view occasions[idx]
      occs = unique(subjoccasion)
      doses = map(occs) do occ
        dose_idx = findfirst(isequal(occ), subjoccasion)
        dose_time = subjtime[dose_idx]
        NCADose(dose_time*timeu, zero(amtu), nothing, DosingUnknown)
      end
    else
      doses = nothing
    end
    try
      ncas[i] = NCASubject(concs[idx], times[idx]; id=id, group=group, dose=doses, concu=concu, timeu=timeu, kwargs...)
    catch
      @info "ID $id errored"
      rethrow()
    end
  end
  # Use broadcast to tighten ncas element type
  pop = NCAPopulation(identity.(ncas))
  added_ii = ii !== nothing && (ii isa Number || (ii isa Array && length(pop) == length(ii)))
  added_ii && add_ii!(pop, ii) # fast path
  return pop, added_ii
end
