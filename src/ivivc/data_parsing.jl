using CSV, DataFrames

# Vivo Data
"""
    read_vivo(df::Union{DataFrame,AbstractString}; id=:id, time=:time,
             conc=:conc, form=:form, dose=:dose, kwargs...)

Parse a `DataFrame` object or a CSV file to `VIVOSubject` or `VIVOPopulation`
which holds an array of `VIVOSubject`s.
"""
read_vivo(file::AbstractString; kwargs...) = read_vivo(CSV.read(file); kwargs...)

function read_vivo(df; group=nothing, kwargs...)
  pop = if group === nothing
    ___read_vivo(df; kwargs...)
  else
    error("not implemented")
  end
  return pop
end

function ___read_vivo(df; id=:id, time=:time, conc=:conc, form=:form, dose=:dose, kwargs...)
  local ids, times, concs, forms, doses
  try
    df[id]
    df[time]
    df[conc]
    df[form]
    df[dose]
  catch x
    @info "The CSV file has keys: $(names(df))"
    throw(x)
  end
  dfnames = names(df)
  sortvars = (id, time)
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sort(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  ids   = df[id]
  times = df[time]
  concs = df[conc]
  forms = df[form]
  doses = df[dose]
  uids = unique(ids)
  idx  = -1
  # FIXME! This is better written as map(uids) do id it currently triggers a dispatch bug in Julia via CSV
  ncas = Vector{Any}(undef, length(uids))
  for (i, id) in enumerate(uids)
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    # the time array for the i-th subject
    subjtime = @view times[idx]
    dose_idx = findall(x->!ismissing(x) && x > zero(x), @view doses[idx])
  
    if length(dose_idx) == 1
      dose_idx = dose_idx[1]
      dose_time = subjtime[dose_idx[1]]
    else
      error("for a subject, dose column should contain only one nonzero value")
    end
      
    try
      ncas[i] = VivoSubject(concs[idx], times[idx], forms[idx], doses[idx], ids[idx], kwargs...)
    catch
      @info "ID $id errored"
      rethrow()
    end
  end
  # Use broadcast to tighten ncas element type
  pop = VivoPopulation(identity.(ncas))
  return pop
end


# Vitro Data
"""
    read_vitro(df::Union{DataFrame,AbstractString}; id=:id, time=:time,
             conc=:conc, form=:form, kwargs...)

Parse a `DataFrame` object or a CSV file to `VitroSubject` or `VitroPopulation`
which holds an array of `VitroSubject`s.
"""
read_vitro(file::AbstractString; kwargs...) = read_vitro(CSV.read(file); kwargs...)

function read_vitro(df; group=nothing, kwargs...)
  pop = if group === nothing
    ___read_vitro(df; kwargs...)
  else
    error("not implemented")
  end
  return pop
end

function ___read_vitro(df; id=:id, time=:time, conc=:conc, form=:form, kwargs...)
  local ids, times, concs, forms
  try
    df[id]
    df[time]
    df[conc]
    df[form]
  catch x
    @info "The CSV file has keys: $(names(df))"
    throw(x)
  end
  dfnames = names(df)
  sortvars = (id, time)
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sort(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  ids   = df[id]
  times = df[time]
  concs = df[conc]
  forms = df[form]
  uids = unique(ids)
  idx  = -1
  # FIXME! This is better written as map(uids) do id it currently triggers a dispatch bug in Julia via CSV
  ncas = Vector{Any}(undef, length(uids))
  for (i, id) in enumerate(uids)
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    try
      ncas[i] = VitroSubject(concs[idx], times[idx], forms[idx], ids[idx], kwargs...)
    catch
      @info "ID $id errored"
      rethrow()
    end
  end
  # Use broadcast to tighten ncas element type
  pop = VitroPopulation(identity.(ncas))
  return pop
end