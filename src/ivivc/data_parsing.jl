# Vivo Data
"""
    read_vivo(df::Union{DataFrame,AbstractString}; id=:id, time=:time,
             conc=:conc, form=:form, dose=:dose, kwargs...)

Parse a `DataFrame` object or a CSV file to `VivoData`
which holds an array of `VivoForm`s.
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

function ___read_vivo(df; id=:id, time=:time, conc=:conc, formulation=:form, dose=:dose, kwargs...)
  local ids, times, concs, forms, doses
  key_check = hasproperty(df, id) && hasproperty(df, time) && hasproperty(df, conc) && hasproperty(df, form) && hasproperty(df, dose)
  if !key_check
    @info "The CSV file has keys: $(names(df))"
    throw(ArgumentError("The CSV file must have: id, time, conc, form, dose"))
  end
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
  uforms = unique(forms)
  idx  = -1
  ncas = Vector{Any}(undef, length(uids))
  for (i, id) in enumerate(uids)
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    ind = Dict{eltype(forms), VivoForm}()
    for form in uforms
      if form in forms[idx]
        idx_n = findfirst(isequal(form), forms[idx]):findlast(isequal(form), forms[idx])
        try
          ind[form] = VivoForm(concs[idx_n], times[idx_n], form, doses[idx_n[1]], id, kwargs...)
        catch
          @info "ID $id errored for formulation $(form)"
          rethrow()
        end
      end
    end
    ncas[i] = ind
  end
  # Use broadcast to tighten ncas element type
  pop = VivoData(identity.(ncas))
  return pop
end


# Vitro Data
"""
    read_vitro(df::Union{DataFrame,AbstractString}; id=:id, time=:time,
             conc=:conc, form=:form, kwargs...)

Parse a `DataFrame` object or a CSV file to `VitroData`
which holds an array of `VitroForm`s.
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
  key_check = hasproperty(df, id) && hasproperty(df, time) && hasproperty(df, conc) && hasproperty(df, form)
  if !key_check
    @info "The CSV file has keys: $(names(df))"
    throw(ArgumentError("The CSV file must have: id, time, conc, form"))
  end
  sortvars = (id, time)
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sort(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  ids   = df[id]
  times = df[time]
  concs = df[conc]
  forms = df[form]
  uids = unique(ids)
  uforms = unique(forms)
  idx  = -1
  ncas = Vector{Any}(undef, length(uids))
  for (i, id) in enumerate(uids)
    # id's range, and we know that it is sorted
    idx = findfirst(isequal(id), ids):findlast(isequal(id), ids)
    ind = Dict{eltype(forms), VitroForm}()
    for form in uforms
      if form in forms[idx]
        idx_n = findfirst(isequal(form), forms[idx]):findlast(isequal(form), forms[idx])
        try
          ind[form] = VitroForm(concs[idx_n], times[idx_n], form, id, kwargs...)
        catch
          @info "ID $id errored for formulation $(form)"
          rethrow()
        end
      end
    end
    ncas[i] = ind
  end
  # Use broadcast to tighten ncas element type
  pop = VitroData(identity.(ncas))
  return pop
end
