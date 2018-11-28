mutable struct NCAdata{C,T,AUC,AUMC,Z,D}
  conc::C
  time::T
  maxidx::Int
  lastidx::Int
  auc_inf::Union{Nothing,AUC}
  auc_last::Union{Nothing,AUC}
  aumc_inf::Union{Nothing,AUMC}
  aumc_last::Union{Nothing,AUMC}
  lambdaz::Union{Nothing,Z}
  dose::Union{Nothing,D}
end

function NCAdata(conc′, time′, dose=nothing; clean=true, kwargs...)
  conc, time = clean ? cleanblq(conc′, time′; kwargs...) : (conc′, time′)
  _, maxidx = conc_maximum(conc, eachindex(conc))
  lastidx = ctlast_idx(conc, time)
  unitconc = oneunit(eltype(conc))
  unittime = oneunit(eltype(time))
  auc_proto = unitconc * unittime
  aumc_proto = auc_proto * unittime
  lambdaz_proto = inv(unittime)
  NCAdata{typeof(conc), typeof(time),
          typeof(auc_proto), typeof(aumc_proto),
          typeof(lambdaz_proto), typeof(dose)}(
            conc, time, maxidx, lastidx,
              ntuple(i->nothing, 6)...)
end

# now, it only shows the types
showunits(nca::NCAdata, args...) = showunits(stdout, nca, args...)
function showunits(io::IO, ::NCAdata{C,T,AUC,AUMC,Z,D}, indent=0) where {C,T,AUC,AUMC,Z,D}
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
