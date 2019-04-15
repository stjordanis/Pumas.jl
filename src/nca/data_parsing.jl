using CSV, DataFrames

"""
  parse_ncadata(df::Union{DataFrame,AbstractString}; id=:ID, time=:time,
    conc=:conc, occasion=nothing, amt=nothing, formulation=nothing, reference=nothing,
    kwargs...) -> NCAPopulation

Parse a `DataFrame` object or a CSV file to `NCAPopulation`. `NCAPopulation`
holds an array of `NCASubject`s which can cache certain results to achieve
efficient NCA calculation.
"""
parse_ncadata(file::AbstractString; kwargs...) = parse_ncadata(CSV.read(file); kwargs...)
# TODO: add ploting time
# TODO: infusion
# TODO: plot time
function parse_ncadata(df; group=nothing, ii=nothing, kwargs...)
  pop, added_ii = if group === nothing
    ___parse_ncadata(df; ii=ii, kwargs...)
  else
    dfs = groupby(df, group)
    groupnum = length(dfs)
    added_ii = true
    dfpops = map(dfs) do df
      if group isa AbstractArray && length(group) > 1
        groupname = map(string, first(df[group]))
        grouplabel = map(string, group)
        currentgroup = join(Base.Iterators.flatten(zip(grouplabel, groupname)), ' ')
      else
        currentgroup = group isa Symbol ? first(df[group]) : first(df[group[1]])
      end
      pop, tmp = ___parse_ncadata(df; group=currentgroup, ii=ii, kwargs...)
      added_ii &= tmp
      return pop
    end
    pops = map(i->dfpops[i][end], 1:groupnum)
    NCAPopulation(vcat(pops...)), added_ii
  end
  ii !== nothing && (added_ii || add_ii!(pop, ii)) # avoiding recomputing the fast path
  return pop
end

function ___parse_ncadata(df; id=:ID, group=nothing, time=:time, conc=:conc, occasion=nothing,
                       amt=nothing, formulation=nothing, route=nothing,# rate=nothing,
                       duration=nothing, ii=nothing, concu=true, timeu=true, amtu=true, warn=true, kwargs...)
  local ids, times, concs, amts, formulations
  try
    df[id]
    df[time]
    df[conc]
    amt === nothing ? nothing : df[amt]
    formulation === nothing ? nothing : df[formulation]
    #rate === nothing ? nothing : df[rate]
    duration === nothing ? nothing : df[duration]
  catch x
    @info "The CSV file has keys: $(names(df))"
    throw(x)
  end
  hasdose = amt !== nothing && formulation !== nothing && route !== nothing
  if warn
    hasdose || @warn "No dosage information has passed. If the dataset has dosage information, you can pass the column names by `amt=:AMT, formulation=:FORMULATION, route=(iv = \"IV\",)`, where `route` can either be `(iv = \"Intravenous\",)`, `(ev = \"Oral\",)` or `(ev = [\"tablet\", \"capsule\"], iv = \"Intra\")`."
    if hasdose && !( route isa NamedTuple && length(route) <= 2 && all(i -> i in (:ev, :iv), keys(route)) )
      throw(ArgumentError("route must be in the form of `(iv = \"Intravenous\",)`, `(ev = \"Oral\",)` or `(ev = [\"tablet\", \"capsule\"], iv = \"Intra\")`. Got $(repr(route))."))
    end
  end
  sortvars = occasion === nothing ? (id, time) : (id, time, occasion)
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sort(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  ids   = df[id]
  times = df[time]
  concs = df[conc]
  amts  = amt === nothing ? nothing : df[amt]
  formulations = formulation === nothing ? nothing : df[formulation]
  occasions = occasion === nothing ? nothing : df[occasion]
  uids = unique(ids)
  idx  = -1
  ncas = map(uids) do id
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
      formulation = let route = route
        map(dose_idx) do i
          f = formulations[i]
          if length(route) == 1
            rt′ = first(route)
            rt = rt′ isa AbstractArray ? rt′ : (rt′,)
            ky = first(keys(route))
            ky === :iv ? (f in rt ? IVBolus : EV) : (f in rt ? EV : IVBolus)
          else # length == 2
            rtiv′ = route[:iv]
            rtev′ = route[:ev]
            rtiv = rtiv′ isa AbstractArray ? rtiv′ : (rtiv′,)
            rtev = rtev′ isa AbstractArray ? rtev′ : (rtev′,)
            f in rtiv && return IVBolus
            f in rtev && return EV
            throw(ArgumentError("$f is not specified to be either EV or IV. Please adjust the `route` argument."))
          end
        end
      end
      duration′ = duration === nothing ? nothing : df[duration][dose_idx]
      doses = NCADose.(dose_time*timeu, amts[dose_idx]*amtu, duration′, formulation)
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
      NCASubject(concs[idx], times[idx]; id=id, group=group, dose=doses, concu=concu, timeu=timeu, kwargs...)
    catch
      @info "ID $id errored"
      rethrow()
    end
  end
  pop = NCAPopulation(ncas)
  added_ii = ii !== nothing && (ii isa Number || (ii isa Array && length(pop) == length(ii)))
  added_ii && add_ii!(pop, ii) # fast path
  return pop, added_ii
end
