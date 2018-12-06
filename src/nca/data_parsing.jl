using CSV, DataFrames

parse_ncadata(file::AbstractString; kwargs...) = parse_ncadata(CSV.read(file); kwargs...)
function parse_ncadata(df::DataFrame; id=:ID, time=:time, conc=:conc, amt=:amt, formulation=:formulation, iv="IV", kwargs...)
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
  uids  = unique(ids)
  idx = -1
  ncas = map(uids) do id
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids) # id's range
    dose_idx = findall(x->!ismissing(x) && x > zero(x), @view amts[idx])
    normalized_dose_idx = dose_idx .- (idx[1] - 1)
    formulation = formulations[dose_idx[1]] == iv ? IV : EV
    doses = NCAdose.(normalized_dose_idx, Ref(formulation), amts[dose_idx])
    NCAdata(concs[idx], times[idx]; dose=doses, kwargs...)
  end
  return ncas
end
