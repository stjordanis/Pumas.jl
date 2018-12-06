using CSV, DataFrames

function parse_ncadata(df::DataFrame; id=:ID, time=:time, conc=:conc, amt=:amt, formulation=:formulation)
  id′   = df[id]
  time′ = df[time]
  conc′ = df[conc]
  amt′  = df[amt]
  formulation′ = df[formulation]

end
