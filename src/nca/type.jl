using Unitful

mutable struct NCAData{C,T,AUC,AUMC,D,Z,F,N}
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

function NCAData(conc′, time′; dose=nothing, llq=nothing, clean=true, lambdaz=nothing, kwargs...)
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
  NCAData{typeof(conc), typeof(time),
          typeof(auc_proto), typeof(aumc_proto),
          typeof(dose), typeof(lambdaz_proto),
          typeof(r2_proto), typeof(llq)}(
            conc, time, maxidx, lastidx, dose, lambdaz, llq,
              ntuple(i->nothing, 6)...)
end

function _auctype(::NCAData{C,T,AUC,AUMC,D,Z,F,N}, auc=:AUCinf) where {C,T,AUC,AUMC,D,Z,F,N}
  if auc in (:AUClast, :AUCinf)
    return AUC
  else
    return AUMC
  end
end

showunits(nca::NCAData, args...) = showunits(stdout, nca, args...)
function showunits(io::IO, ::NCAData{C,T,AUC,AUMC,D,Z,F,N}, indent=0) where {C,T,AUC,AUMC,D,Z,F,N}
  pad   = " "^indent
  if eltype(C) <: Quantity
    conct = string(unit(eltype(C)))
    timet = string(unit(eltype(T)))
    AUCt  = string(unit(AUC))
    AUMCt = string(unit(AUMC))
    Zt    = string(unit(Z))
    Dt    = D === Nothing ? D : string(unit(D))
  else
    conct = nameof(eltype(C))
    timet = nameof(eltype(T))
    AUCt  = AUC
    AUMCt = AUMC
    Zt    = Z
    Dt    = D
  end
  println(io, "$(pad)concentration: $conct")
  println(io, "$(pad)time:          $timet")
  println(io, "$(pad)auc:           $AUCt")
  println(io, "$(pad)aumc:          $AUMCt")
  println(io, "$(pad)λz:            $Zt")
  print(  io, "$(pad)dose:          $Dt")
end

function Base.show(io::IO, nca::NCAData)
  println(io, "NCAData:")
  showunits(io, nca, 2)
end

# Summary and report
struct NCAReport{S,V}
  settings::S
  values::V
end

function NCAReport(nca::NCAData{C,T,AUC,AUMC,D,Z,F,N}; kwargs...) where {C,T,AUC,AUMC,D,Z,F,N}
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
  
  NCAReport(settings, values)
end

Base.summary(::NCAReport) = "NCAReport"
function Base.show(io::IO, ::MIME"text/plain", report::NCAReport)
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
