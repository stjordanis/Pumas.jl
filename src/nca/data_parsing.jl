using CSV, DataFrames

parse_ncadata(file::AbstractString; kwargs...) = parse_ncadata(CSV.read(file); kwargs...)
function parse_ncadata(df::DataFrame; id=:ID, time=:time, conc=:conc, occasion=nothing,
                       amt=:amt, formulation=:formulation, iv="IV", kwargs...)
  local ids, times, concs, amts, formulations
  try
    ids   = df[id]
    times = df[time]
    concs = df[conc]
    amts  = df[amt]
    formulations = df[formulation]
  catch x
    @info "The CSV file has keys: $(names(df))"
    throw(x)
  end
  sortvars = occasion === nothing ? id : (id, occasion)
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sorted(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  uids = unique(ids)
  idx  = -1
  ncas = map(uids) do id
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    # we already sorted by occasions, so we don't have to think about it now.
    dose_idx = findall(x->!ismissing(x) && x > zero(x), @view amts[idx])
    length(dose_idx) > 1 && occasion === nothing && error("`occasion` must be provided for multiple dosing data")
    # We want to use time instead of an integer index here, because later we
    # need to remove BLQ and missing data, so that an index number will no
    # longer be valid.
    #
    # the time array for the i-th subject
    subjtime = @view times[idx]
    if length(dose_idx) == 1
      dose_idx = dose_idx[1]
      dose_time = subjtime[dose_idx[1]]
    else
      dose_time = similar(times, Base.nonmissingtype(eltype(times)), length(dose_idx))
      for (n,i) in enumerate(dose_idx)
        dose_time[n] = subjtime[i]
      end
    end
    formulation = formulations[dose_idx[1]] == iv ? IV : EV
    doses = NCADose.(dose_time, Ref(formulation), amts[dose_idx])
    NCASubject(concs[idx], times[idx]; id=id, dose=doses, kwargs...)
  end
  return NCAPopulation(ncas)
end
