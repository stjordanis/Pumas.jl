using CSV, DataFrames

"""
  parse_ncadata(df::DataFrame; id=:ID, time=:time, conc=:conc, occasion=nothing,
                       amt=nothing, formulation=nothing, iv=nothing, kwargs...)

Parse a dataframe object  an external spredsheet file as an NCA dataset.

The resulting object has the following properties
TODO: Yingbo - can we fill in details here please
"""
parse_ncadata(file::AbstractString; kwargs...) = parse_ncadata(CSV.read(file); kwargs...)
function parse_ncadata(df::DataFrame; id=:ID, time=:time, conc=:conc, occasion=nothing,
                       amt=nothing, formulation=nothing, iv=nothing, kwargs...)
  local ids, times, concs, amts, formulations
  try
    ids   = df[id]
    times = df[time]
    concs = df[conc]
    amts  = amt === nothing ? nothing : df[amt]
    formulations = formulation === nothing ? nothing : df[formulation]
  catch x
    @info "The CSV file has keys: $(names(df))"
    throw(x)
  end
  hasdose = !(amts === nothing) && !(formulation === nothing) && !(iv === nothing)
  if !hasdose
    @warn "No dosage information has passed. If the dataset has dosage information, you can pass the column names by
    `amt=:AMT, formulation=:FORMULATION, iv=\"IV\"`"
  end
  sortvars = occasion === nothing ? id : (id, occasion)
  # I think sortvars should be
  # sortvars = occasion === nothing ? (id,time) : (id, time, occasion)
  # but in order to sort with time, we need to make sure there is no missing time
  # or non-numeric time. Not sure how you will implement that
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sorted(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  uids = unique(ids)
  idx  = -1
  ncas = map(uids) do id
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    # the time array for the i-th subject
    subjtime = @view times[idx]
    # we already sorted by occasions, so we don't have to think about it now.
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
      formulation = map(i -> formulations[i] == iv ? IV : EV, dose_idx)
      doses = NCADose.(dose_time, formulation, amts[dose_idx])
    else
      doses = nothing
    end
    NCASubject(concs[idx], times[idx]; id=id, dose=doses, kwargs...)
  end
  return NCAPopulation(ncas)
end
