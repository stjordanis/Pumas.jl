mutable struct NCAdata{C,T,AUC,AUMC,D,Z,F,N}
  conc::C
  time::T
  maxidx::Int
  lastidx::Int
  dose::Union{Nothing,D}
  lambdaz::Union{Nothing,Z}
  llq::N
  r2::Union{Nothing,F}
  points::Union{Nothing,Int}
  auc_inf::Union{Nothing,AUC}
  auc_last::Union{Nothing,AUC}
  aumc_inf::Union{Nothing,AUMC}
  aumc_last::Union{Nothing,AUMC}
end

function NCAdata(conc′, time′; dose=nothing, llq=nothing, clean=true, lambdaz=nothing, kwargs...)
  conc, time = clean ? cleanblq(conc′, time′; llq=llq, kwargs...) : (conc′, time′)
  _, maxidx = conc_maximum(conc, eachindex(conc))
  lastidx = ctlast_idx(conc, time; llq=llq, check=false)
  unitconc = oneunit(eltype(conc))
  unittime = oneunit(eltype(time))
  llq === nothing && (llq = unitconc*false)
  auc_proto = unitconc * unittime
  aumc_proto = auc_proto * unittime
  r2_proto = unittime/unitconc*false
  # TODO: check units
  lambdaz_proto = inv(unittime)
  NCAdata{typeof(conc), typeof(time),
          typeof(auc_proto), typeof(aumc_proto),
          typeof(dose), typeof(lambdaz_proto),
          typeof(r2_proto), typeof(llq)}(
            conc, time, maxidx, lastidx, dose, lambdaz, llq,
              ntuple(i->nothing, 6)...)
end

function _auctype(::NCAdata{C,T,AUC,AUMC,D,Z,F,N}, auc=:AUCinf) where {C,T,AUC,AUMC,D,Z,F,N}
  if auc in (:AUClast, :AUCinf)
    return AUC
  else
    return AUMC
  end
end

# now, it only shows the types
showunits(nca::NCAdata, args...) = showunits(stdout, nca, args...)
function showunits(io::IO, ::NCAdata{C,T,AUC,AUMC,D,Z,F,N}, indent=0) where {C,T,AUC,AUMC,D,Z,F,N}
  conct = nameof(eltype(C))
  timet = nameof(eltype(T))
  pad   = " "^indent
  println(io, "$(pad)concentration: $conct")
  println(io, "$(pad)time:          $timet")
  println(io, "$(pad)auc:           $AUC")
  println(io, "$(pad)aumc:          $AUMC")
  println(io, "$(pad)λz:            $Z")
  print(  io, "$(pad)dose:          $D")
end

function Base.show(io::IO, nca::NCAdata)
  println(io, "NCAdata:")
  showunits(io, nca, 2)
end

# Summary and report
struct NCAreport{S,V}
  settings::S
  values::V
end

function NCAreport(nca::NCAdata{C,T,AUC,AUMC,D,Z,F,N}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N}
  D === Nothing && @warn "`dose` is not provided. No dependent quantities will be calculated"
  # TODO: Summarize settings
  settings = nothing

  # Calculate summary values
  clast′ = clast(nca; kwargs...)
  tlast′ = tlast(nca; kwargs...)
  tmax′  = tmax(nca; kwargs...)
  cmax′  = cmax(nca; kwargs...)
  λz     = lambdaz(nca; kwargs...)[1]
  thalf′ = thalf(nca; kwargs...)
  auc′   = auc(nca; kwargs...)
  aucp   = auc_extrap_percent(nca; kwargs...)
  aumc′  = aumc(nca; kwargs...)
  aumcp  = aumc_extrap_percent(nca; kwargs...)
  values = (clast=clast′, tlast=tlast′, tmax=tmax′, cmax=cmax′,
   lambdaz=λz, thalf=thalf′, auc=auc′, auc_extrap_percent=aucp,
   aumc=aumc′, aumc_extrap_percent=aumcp)
  
  NCAreport(settings, values)
end

Base.summary(::NCAreport) = "NCAreport"
function Base.show(io::IO, ::MIME"text/plain", report::NCAreport)
  println(io, "# Noncompartmental Analysis Report")
  println(io, "PuMaS version " * string(Pkg.installed()["PuMaS"]))
  println(io, "Julia version " * string(VERSION))
  println(io, "")
  println(io, "Date and Time: " * Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
  println(io, "")

  println(io, "## Calculation Setting")
  # TODO
  println(io, "")

  println(io, "## Calculated Values")
  println(io, "|         Name         |    Value   |")
  println(io, "|----------------------|------------|")
  for entry in pairs(report.values)
    @printf(io, "| %-20s | %10f |\n", entry[1], entry[2])
  end
end
