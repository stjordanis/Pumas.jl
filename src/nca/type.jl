mutable struct NCAdata{C,T,AUC,AUMC,D,Z,F}
  conc::C
  time::T
  maxidx::Int
  lastidx::Int
  dose::Union{Nothing,D}
  lambdaz::Union{Nothing,Z}
  r2::Union{Nothing,F}
  points::Union{Nothing,Int}
  auc_inf::Union{Nothing,AUC}
  auc_last::Union{Nothing,AUC}
  aumc_inf::Union{Nothing,AUMC}
  aumc_last::Union{Nothing,AUMC}
end

function NCAdata(conc′, time′; dose=nothing, clean=true, lambdaz=nothing,
                 llq=nothing, kwargs...)
  conc, time = clean ? cleanblq(conc′, time′; kwargs...) : (conc′, time′)
  _, maxidx = conc_maximum(conc, eachindex(conc))
  lastidx = ctlast_idx(conc, time, llq=llq, check=false)
  unitconc = oneunit(eltype(conc))
  unittime = oneunit(eltype(time))
  auc_proto = unitconc * unittime
  aumc_proto = auc_proto * unittime
  r2_proto = unittime/unitconc*false
  # TODO: check units
  lambdaz_proto = inv(unittime)
  NCAdata{typeof(conc), typeof(time),
          typeof(auc_proto), typeof(aumc_proto),
          typeof(dose), typeof(lambdaz_proto),
          typeof(r2_proto)}(
            conc, time, maxidx, lastidx, dose, lambdaz,
              ntuple(i->nothing, 6)...)
end

# now, it only shows the types
showunits(nca::NCAdata, args...) = showunits(stdout, nca, args...)
function showunits(io::IO, ::NCAdata{C,T,AUC,AUMC,D,Z,F}, indent=0) where {C,T,AUC,AUMC,D,Z,F}
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
