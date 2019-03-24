using CSV, DataFrames

"""
  parse_ncadata(df::Union{DataFrame,AbstractString}; id=:ID, time=:time,
    conc=:conc, occasion=nothing, amt=nothing, formulation=nothing, iv=nothing,
    kwargs...) -> NCAPopulation

Parse a `DataFrame` object or a CSV file to `NCAPopulation`. `NCAPopulation`
holds an array of `NCASubject`s which can cache certain results to achieve
efficient NCA calculation.
"""
parse_ncadata(file::AbstractString; kwargs...) = parse_ncadata(CSV.read(file); kwargs...)
# TODO: add ploting time
# TODO: infusion
# TODO: plot time
function parse_ncadata(df; group=nothing, kwargs...)
  if group === nothing
    ___parse_ncadata(df; kwargs...)
  else
    dfs = groupby(df, group)
    dfpops = map(dfs) do df
      ___parse_ncadata(df; kwargs...)
    end
    groups = map(i->first(dfpops[i][1]), 1:length(dfpops))
    pops = map(i->dfpops[i][2], 1:length(dfpops))
    NCAPopulation(vcat(pops...))
  end
end

function ___parse_ncadata(df; id=:ID, time=:time, conc=:conc, occasion=nothing,
                       amt=nothing, formulation=nothing, iv=nothing,
                       concu=true, timeu=true, amtu=true, warn=true, kwargs...)
  local ids, times, concs, amts, formulations
  try
    df[id]
    df[time]
    df[conc]
    amt === nothing ? nothing : df[amt]
    formulation === nothing ? nothing : df[formulation]
  catch x
    @info "The CSV file has keys: $(names(df))"
    throw(x)
  end
  hasdose = !(amt === nothing) && !(formulation === nothing) && !(iv === nothing)
  if warn && !hasdose
    @warn "No dosage information has passed. If the dataset has dosage information, you can pass the column names by
    `amt=:AMT, formulation=:FORMULATION, iv=\"IV\"`"
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
      formulation = map(i -> formulations[i] == iv ? IVBolus : EV, dose_idx)
      doses = NCADose.(dose_time*timeu, amts[dose_idx]*amtu, formulation)
    elseif occasion !== nothing
      subjoccasion = @view occasions[idx]
      occs = unique(subjoccasion)
      doses = map(occs) do occ
        dose_idx = findfirst(isequal(occ), subjoccasion)
        dose_time = subjtime[dose_idx]
        NCADose(dose_time*timeu, zero(amtu), DosingUnknown)
      end
    else
      doses = nothing
    end
    try
      NCASubject(concs[idx], times[idx]; id=id, dose=doses, concu=concu, timeu=timeu, kwargs...)
    catch
      @info "ID $id errored"
      rethrow()
    end
  end
  return NCAPopulation(ncas)
end
