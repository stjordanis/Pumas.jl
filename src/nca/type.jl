using Unitful
using Markdown

"""
    Formulation

Type of formulations. There are IV (intravenous) and EV (extravascular).
"""
@enum Formulation IVBolus IVInfusion EV# DosingUnknown
# Formulation behaves like scalar
Broadcast.broadcastable(x::Formulation) = Ref(x)

"""
    NCADose

`NCADose` takes the following arguments
- `time`: time of the dose
- `amt`: The amount of dosage
- `formulation`: Type of formulation, `NCA.IVBolus`, `NCA.IVInfusion` or `NCA.EV`
- `ii`: interdose interval
- `ss`: steady-state
"""
struct NCADose{T,A}
  time::T
  amt::A
  duration::T
  formulation::Formulation
  ii::T
  ss::Bool
  function NCADose(time, amt, duration::D, formulation, ii=zero(time), ss=false) where D
    duration′ = D === Nothing ? zero(time) : duration
    formulation′ = formulation === EV ? EV : iszero(duration′) ? IVBolus : IVInfusion
    time, ii = promote(time, ii)
    ii < zero(ii) && throw(ArgumentError("ii must be non-negative. Got ii=$ii"))
    if ii <= zero(ii) && ss
      throw(ArgumentError("ii must be greater than zero when ss=1. Got ii=$ii"))
    end
    return new{typeof(time), typeof(amt)}(time, amt, duration′, formulation′, ii, ss)
  end
end

# NCADose should behave like a scalar in broadcast
Broadcast.broadcastable(x::NCADose) = Ref(x)
Base.first(x::NCADose) = x

function Base.show(io::IO, n::NCADose)
  println(io, "NCADose:")
  println(io, "  time:         $(n.time)")
  println(io, "  amt:          $(n.amt)")
  println(io, "  duration:     $(n.duration)")
  println(io, "  formulation:  $(n.formulation)")
  print(  io, "  ss:           $(n.ss)")
end

# any changes in here must be reflected to ./simple.jl, too
mutable struct NCASubject{C,T,TT,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
  # allow id to be non-int, e.g. 001-100-1001 or STUD011001
  id::ID
  group::G
  conc::C
  rate::R
  # `time` will be time after dose (TAD) for multiple dosing, and it will be in
  # the form of [TAD₁, TAD₂, ..., TADₙ] where `TADᵢ` denotes the TAD for the
  # `i`th dose, which is a vector. Hence, `time` could be a vector of vectors.
  time::T
  start_time::Union{Nothing, T}
  end_time::Union{Nothing, T}
  volume::V
  # `abstime` is always a vector, and it leaves untouched except BLQ handling
  abstime::TT
  maxidx::I
  lastidx::I
  dose::D
  lambdaz::Z
  llq::N
  r2::F
  adjr2::F
  intercept::F
  firstpoint::tEltype
  lastpoint::tEltype
  points::P
  auc_last::AUC
  auc_0::AUC
  aumc_last::AUMC
  method::Symbol
end

"""
    NCASubject(conc, time; concu=true, timeu=true, id=1, group=nothing, dose::T=nothing, llq=nothing, lambdaz=nothing, clean=true, check=true, kwargs...)

Constructs a NCASubject
"""
function NCASubject(conc, time;
                    start_time=nothing, end_time=nothing, volume=nothing, concu=true, timeu=true, volumeu=true,
                    id=1, group=nothing, dose::T=nothing, llq=nothing,
                    lambdaz=nothing, clean=true, check=true, kwargs...) where T
  time isa AbstractRange && (time = collect(time))
  conc isa AbstractRange && (conc = collect(conc))
  if concu !== true
    conc = map(x -> x === missing ? x : x*concu, conc)
  end
  if timeu !== true
    if time !== nothing
      time = map(x -> x === missing ? x : x*timeu, time)
    end
    if start_time !== nothing
      start_time = map(x -> x === missing ? x : x*timeu, start_time)
    end
    if end_time !== nothing
      end_time = map(x -> x === missing ? x : x*timeu, end_time)
    end
  end
  multidose = T <: AbstractArray && length(dose) > 1
  nonmissingeltype(x) = Base.nonmissingtype(eltype(x))
  unitconc = float(oneunit(nonmissingeltype(conc)))
  unittime = float(oneunit(nonmissingeltype(time)))
  llq === nothing && (llq = zero(unitconc))
  auc_proto = -unitconc * unittime
  aumc_proto = auc_proto * unittime
  intercept = r2_proto = ustrip(unittime/unitconc)
  _lambdaz = inv(unittime)
  lambdaz_proto = lambdaz === nothing ? _lambdaz : lambdaz
  if multidose
    n = length(dose)
    abstime = clean ? typeof(unittime)[] : time
    ct = let time=time, dose=dose
      map(eachindex(dose)) do i
        idxs = ithdoseidxs(time, dose, i)
        conci, timei = @view(conc[idxs]), @view(time[idxs])
        check && checkconctime(conci, timei; dose=dose, kwargs...)
        if clean
          conci, timei, _, _ = cleanmissingconc(conci, timei; kwargs...)
        end
        conc′, time′ = clean ? cleanblq(conci, timei; llq=llq, dose=dose, kwargs...)[1:2] : (conc[idxs], time[idxs])
        clean && append!(abstime, time′)
        iszero(time[idxs[1]]) ? (conc′, time′) : (conc′, time′.-time[idxs[1]]) # convert to TAD
      end
    end
    conc = map(x->x[1], ct)
    time = map(x->x[2], ct)
    maxidx  = fill(-2, n)
    lastidx = @. ctlast_idx(conc, time; llq=llq)
    return NCASubject{typeof(conc), typeof(time), typeof(abstime), eltype(time),
                      Vector{typeof(auc_proto)}, Vector{typeof(aumc_proto)},
                      typeof(dose), Vector{typeof(lambdaz_proto)},
                      Vector{typeof(r2_proto)}, typeof(llq), typeof(lastidx),
                      Vector{Int}, typeof(id), typeof(group), Nothing, Nothing}(id, group,
                        conc, nothing, time, #= no multidose urine =# nothing, nothing, nothing, abstime,
                        maxidx, lastidx, dose, fill(lambdaz_proto, n), llq,
                        fill(r2_proto, n), fill(r2_proto, n), fill(intercept, n),
                        fill(-unittime, n), fill(-unittime, n), fill(0, n),
                        fill(auc_proto, n), fill(auc_proto, n), fill(aumc_proto, n), :___)
  end
  isurine = volume !== nothing
  volume = isurine ? volume .* volumeu : nothing
  if check
    if isurine
      checkconctime(conc, start_time; dose=dose, kwargs...)
      checkconctime(conc, end_time;   dose=dose, kwargs...)
    else
      checkconctime(conc, time; dose=dose, kwargs...)
    end
  end
  if clean
    if isurine
      conc, start_time, end_time, volume = cleanmissingconc(conc, start_time; end_time=end_time, volume=volume, kwargs...)
      conc, start_time, end_time, volume = cleanblq(conc, start_time; end_time=end_time, volume=volume, llq=llq, dose=dose, kwargs...)
    else
      conc, time, _, _ = cleanmissingconc(conc, time; kwargs...)
      conc, time, _, _ = cleanblq(conc, time; llq=llq, dose=dose, kwargs...)
    end
  end
  if isurine
    Δt = @. end_time - start_time
    time = @. start_time + Δt/2
    rate = @.(volume*conc/Δt)
    auc_proto = -oneunit(eltype(rate)) * oneunit(eltype(time))
  else
    rate = nothing
  end
  abstime = time
  dose !== nothing && (dose = first(dose))
  maxidx = -2
  lastidx = ctlast_idx(conc, time; llq=llq)
  NCASubject{typeof(conc),   typeof(time), typeof(abstime), eltype(time),
          typeof(auc_proto), typeof(aumc_proto),
          typeof(dose),      typeof(lambdaz_proto),
          typeof(r2_proto),  typeof(llq),   typeof(lastidx),
          Int, typeof(id),   typeof(group), typeof(volume), typeof(rate)}(id, group,
            conc, rate, time, start_time, end_time, volume, abstime, maxidx, lastidx, dose, lambdaz_proto, llq,
            r2_proto,  r2_proto, intercept, unittime, unittime, 0,
            auc_proto, auc_proto, aumc_proto, :___)
end

showunits(nca::NCASubject, args...) = showunits(stdout, nca, args...)
function showunits(io::IO, ::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}, indent=0) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
  pad   = " "^indent
  if D <: NCADose
    doseT = D.parameters[2]
  elseif D <: AbstractArray
    doseT = eltype(D).parameters[2]
  else
    doseT = D
  end
  multidose = D <: AbstractArray
  _C = multidose ? eltype(C) : C
  _T = multidose ? eltype(T) : T
  _AUC = multidose ? eltype(AUC) : AUC
  _AUMC = multidose ? eltype(AUMC) : AUMC
  _Z = multidose ? eltype(Z) : Z
  if eltype(_C) <: Quantity
    conct = string(unit(eltype(_C)))
    timet = string(unit(eltype(_T)))
    AUCt  = string(unit(_AUC))
    AUMCt = string(unit(_AUMC))
    Zt    = string(unit(_Z))
    Dt    = D === Nothing ? D : string(unit(doseT))
  else
    conct = nameof(eltype(_C))
    timet = nameof(eltype(_T))
    AUCt  = _AUC
    AUMCt = _AUMC
    Zt    = _Z
    Dt    = doseT
  end
  println(io, "$(pad)concentration: $conct")
  println(io, "$(pad)time:          $timet")
  println(io, "$(pad)auc:           $AUCt")
  println(io, "$(pad)aumc:          $AUMCt")
  println(io, "$(pad)λz:            $Zt")
  print(  io, "$(pad)dose:          $Dt")
end

function Base.show(io::IO, n::NCASubject)
  println(io, "NCASubject:")
  println(io, "  ID: $(n.id)")
  n.group === nothing || println(io, "  Group: $(n.group)")
  pad = 4
  showunits(io, n, pad)
end

@noinline dosenumerror() = throw(AssertionError("All subjects in a population must have the same number of doses. Please consider padding your data if you are sure that there are no errors in it."))
struct NCAPopulation{T<:NCASubject} <: AbstractVector{T}
  subjects::Vector{T}
  function NCAPopulation(_pop::Vector{<:NCASubject})
    E = eltype(_pop)
    isconcretetype(E) || dosenumerror()
    if ismultidose(E)
      ndose = length(_pop[1].dose)
      for subj in _pop
        length(subj.dose) == ndose && continue
        dosenumerror()
      end
    end
    return new{E}(_pop)
  end
end

Base.size(n::NCAPopulation) = size(n.subjects)
function Base.getindex(n::NCAPopulation, I...)
  subj = getindex(n.subjects, I...)
  subj isa Vector ? NCAPopulation(subj) : subj
end
Base.setindex!(n::NCAPopulation, X, I...) = setindex!(n.subjects, X, I...)
Base.show(io::IO, pop::NCAPopulation) = show(io, MIME"text/plain"(), pop)
function Base.show(io::IO, ::MIME"text/plain", pop::NCAPopulation)
  ids = unique([subj.id for subj in pop])
  println(io, "NCAPopulation ($(length(ids)) subjects):")
  println(io, "  ID: $(ids)")
  first(pop).group === nothing || println(io, "  Group: $(unique([subj.group for subj in pop]))")
  showunits(io, first(pop), 4)
end

Base.@pure ismultidose(nca::NCASubject) = ismultidose(typeof(nca))
Base.@pure ismultidose(nca::NCAPopulation) = ismultidose(eltype(nca.subjects))
Base.@pure function ismultidose(::Type{NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}}) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
  return D <: AbstractArray
end

hasdose(nca::NCASubject) = hasdose(typeof(nca))
hasdose(nca::NCAPopulation) = hasdose(eltype(nca.subjects))
function hasdose(::Type{NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}}) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
  return D !== Nothing
end

# Summary and report
struct NCAReport{S,V}
  settings::S
  values::V
end

macro defkwargs(sym, expr...)
  :((nca; kwargs...) -> $sym(nca; $(expr...), kwargs...)) |> esc
end

NCAReport(nca::NCASubject; kwargs...) = NCAReport(NCAPopulation([nca]); kwargs...)
function NCAReport(pop::NCAPopulation; pred=nothing, normalize=nothing, auctype=nothing, # strips out unnecessary user options
                   sigdigits=nothing, kwargs...)
  !hasdose(pop) && @warn "`dose` is not provided. No dependent quantities will be calculated"
  settings = Dict(:multidose=>ismultidose(pop), :subject=>(pop isa NCASubject))

  has_ev = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.formulation === EV :
                            any(d->d.formulation === EV, subj.dose)
  end
  has_iv = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.formulation === IVBolus :
                            any(d->d.formulation === IVBolus, subj.dose)
  end
  has_inf = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.formulation === IVInfusion :
                            any(d->d.formulation === IVInfusion, subj.dose)
  end
  has_ii = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.ii > zero(subj.dose.ii) :
                            any(d->d.ii > zero(d.ii), subj.dose)
  end
  is_ss = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.ss :
                            any(d->d.ss, subj.dose)
  end
  has_iv_inf = has_iv || has_inf
  is_urine = any(subj->subj.volume !== nothing, pop)
  #TODO:
  # add partial AUC
  # interval Cmax??? low priority
  if is_urine
    report_pairs = [
           "dose" => doseamt,
           "urine_volume" => urine_volume,
           "lambda_z" => lambdaz,
           "tmax_rate" => tmax_rate,
           "tlag" => tlag,
           "max_rate" => max_rate,
           "rate_last" => rate_last,
           "mid_time_last" => mid_time_last,
           "amount_recovered" => amount_recovered,
           "percent_recovered" => percent_recovered,
           "aurc_last_obs" => @defkwargs(aurc, auctype=:last),
           "aurc_inf_obs" => aurc,
           "aurc_extrap_obs" => aurc_extrap_percent,
           "rate_last_pred" => @defkwargs(rate_last, pred=true),
           "aurc_inf_pred" => @defkwargs(aurc, pred=true),
           "aurc_extrap_pred" => @defkwargs(aurc_extrap_percent, pred=true),
           "aurc_last_d_obs" => @defkwargs(aurc, auctype=:last, normalize=true),
           "n_samples"          =>     n_samples,
           "rsq"                =>     lambdazr2,
           "rsq_adjusted"       =>     lambdazadjr2,
           "corr_xy"            =>     lambdazr,
           "no_points_lambda_z" =>     lambdaznpoints,
           "lambda_z_intercept" =>     lambdazintercept,
           "lambda_z_lower"     =>     lambdaztimefirst,
           "lambda_z_upper"     =>     lambdaztimelast,
           "span"               =>     span,
           "route"              =>     dosetype,
      ]
  else
    report_pairs = [
           "dose"               =>     doseamt,
           "lambda_z"           =>     lambdaz,
           "half_life"          =>     thalf,
           "tmax"               =>     tmax,
           has_ev && "tlag"     =>     tlag,
           "cmax"               =>     cmax,
           has_ii && "cmaxss"   =>     cmaxss,
           has_iv && "c0"       =>     c0,
           "clast"              =>     clast,
           "clast_pred"         =>     @defkwargs(clast, pred=true),
           "auclast"            =>     auclast,
           "tlast"              =>     tlast,
           is_ss  || "aucinf_obs"   =>     auc,
           has_ii && "auc_tau_obs"  =>     auctau,
           has_iv_inf && "vz_obs"   =>     _vz,
           has_iv_inf && "cl_obs"   =>     _cl,
           has_ev && "vz_f_obs"     =>     _vzf,
           has_ev && "cl_f_obs"     =>     _clf,
           is_ss  || "aucinf_pred"  =>     @defkwargs(auc, pred=true),
           has_ii && "auc_tau_pred" =>     @defkwargs(auctau, pred=true),
           has_iv_inf && "vz_pred"  =>     @defkwargs(_vz, pred=true),
           has_iv_inf && "cl_pred"  =>     @defkwargs(_cl, pred=true),
           has_ev && "vz_f_pred"    =>     @defkwargs(_vzf, pred=true),
           has_ev && "cl_f_pred"    =>     @defkwargs(_clf, pred=true),
           has_iv_inf && "vss_obs"  => vss,
           has_iv_inf && "vss_pred" => @defkwargs(vss, pred=true),
           ((has_inf || has_ev) && is_ss) || has_iv && "tmin"         =>     tmin,
           ((has_inf || has_ev) && is_ss) || has_iv && "cmin"         =>     cmin,
           ((has_inf || has_ev) && is_ss) || has_iv && "cminss"       =>     cminss,
           has_ii && "ctau"         =>     ctau,
           has_ii && "cavgss"       =>     cavgss,
           has_ii && "fluctuation"  =>     fluctuation,
           has_ii && "fluctuation_tau"    =>     @defkwargs(fluctuation, usetau=true),
           has_ii && "accumulation_index" =>     accumulationindex,
           "cmax_d"                 =>     @defkwargs(cmax, normalize=true),
           "auclast_d"              =>     @defkwargs(auclast, normalize=true),
           has_ii || "aucinf_d_obs" =>     @defkwargs(auc, normalize=true),
           "auc_extrap_obs"         =>     auc_extrap_percent,
           has_iv && "auc_back_extrap_obs" => auc_back_extrap_percent,
           has_ii || "aucinf_d_pred"       =>     @defkwargs(auc, normalize=true, pred=true),
           "auc_extrap_pred"        =>     @defkwargs(auc_extrap_percent, pred=true),
           has_iv && "auc_back_extrap_pred" => @defkwargs(auc_back_extrap_percent, pred=true),
           "aumclast"               =>     aumclast,
           "aumcinf_obs"            =>     aumc,
           "aumc_extrap_obs"        =>     aumc_extrap_percent,
           "aumcinf_pred"           =>     @defkwargs(aumc, pred=true),
           "aumc_extrap_pred"       =>     @defkwargs(aumc_extrap_percent, pred=true),
           has_iv_inf && "mrtlast"      =>     @defkwargs(mrt, auctype=:last),
           has_iv_inf && "mrtinf_obs"   =>     @defkwargs(mrt, auctype=:inf),
           has_iv_inf && "mrtinf_pred"  =>     @defkwargs(mrt, auctype=:inf, pred=true),
           has_ii && "swing"        =>     swing,
           has_ii && "swing_tau"    =>     @defkwargs(swing, usetau=true),
           "n_samples"          =>     n_samples,
           "rsq"                =>     lambdazr2,
           "rsq_adjusted"       =>     lambdazadjr2,
           "corr_xy"            =>     lambdazr,
           "no_points_lambda_z" =>     lambdaznpoints,
           "lambda_z_intercept" =>     lambdazintercept,
           "lambda_z_lower"     =>     lambdaztimefirst,
           "lambda_z_upper"     =>     lambdaztimelast,
           "span"               =>     span,
           "route"              =>     dosetype,
           has_ii && "tau"      =>     tau,
        ]
  end
  deleteat!(report_pairs, findall(x->x.first isa Bool, report_pairs))
  _names = map(x->Symbol(x.first), report_pairs)
  funcs = map(x->x.second, report_pairs)
  vals  = [f(pop; label = i == 1, kwargs...) for (i, f) in enumerate(funcs)]
  if sigdigits !== nothing
    for val in vals
      col = val[!,end]
      map!(col, col) do x
        x isa Number ? round(ustrip(x), sigdigits=sigdigits)*oneunit(x) : x
      end
    end
  end
  values = [i == 1 ? val : rename!(val, names(val)[1]=>name) for (i, (val, name)) in enumerate(zip(vals, _names))]

  NCAReport(settings, values)
end

Base.summary(::NCAReport) = "NCAReport"

function Base.show(io::IO, report::NCAReport)
  println(io, summary(report))
  print(io, "  keys: $(map(x->names(x)[1], report.values))")
end

# units go under the header
to_dataframe(report::NCAReport) = convert(DataFrame, report)
function Base.convert(::Type{DataFrame}, report::NCAReport)
  #report.settings[:subject] && return DataFrame(map(x->[x], report.values))
  hcat(report.values...)
end

to_markdown(report::NCAReport) = convert(Markdown.MD, report)
function Base.convert(::Type{Markdown.MD}, report::NCAReport)
  _io = IOBuffer()
  println(_io, "# Noncompartmental Analysis Report")
  println(_io, "Pumas version " * string(Pkg.installed()["Pumas"]))
  println(_io, "Julia version " * string(VERSION))
  println(_io, "")
  println(_io, "Date and Time: " * Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
  println(_io, "")

  println(_io, "## Calculation Setting")
  # TODO
  println(_io, "")

  println(_io, "## Calculated Values")
  println(_io, "|         Name         |    Value   |")
  println(_io, "|:---------------------|-----------:|")
  for entry in report.values
    _name = string(names(entry)[1])
    name = replace(_name, "_"=>"\\_") # escape underscore
    entry[end] === missing && (@printf(_io, "| %s | %s |\n", name, "missing"); continue)
    for v in entry[end]
      val =  v isa Number ? round(ustrip(v), digits=2)*oneunit(v) :
             v
      @printf(_io, "| %s | %s |\n", name, val)
    end
  end
  return Markdown.parse(String(take!(_io)))
end

@recipe function f(subj::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}; linear=true, loglinear=true) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
  subj = urine2plasma(subj)
  istwoplots = linear && loglinear
  istwoplots && (layout --> (1, 2))
  hastitle = length(get(plotattributes, :title, [])) >= 2
  hasunits = eltype(C) <: Quantity
  isurine  = R !== Nothing
  timename = isurine ? "Midpoint" : "Time"
  hasunits && (timename *= " ($(string(unit(eltype(T)))))")
  concname = isurine ? "Excretion Rate" : "Concentration"
  hasunits && (concname *= " ($(string(unit(eltype(C)))))")
  xguide --> timename
  yguide --> concname
  label --> subj.id
  vars = (ustrip.(subj.abstime), replace!(x->x<eps(one(x)) ? eps(one(x)) : x, vcat(map(x->ustrip.(x), subj.conc)...)))
  if linear
    @series begin
      seriestype --> :path
      istwoplots && (subplot --> 1)
      _title = hastitle ? plotattributes[:title][1] : "Linear view"
      title := _title
      vars
    end
  end
  if loglinear
    @series begin
      yscale --> :log10
      seriestype --> :path
      istwoplots && (subplot --> 2)
      _title = hastitle ? plotattributes[:title][2] : "Semilogrithmic view"
      title := _title
      vars
    end
  end
end

@recipe function f(pop::NCAPopulation{NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}}; linear=true, loglinear=true) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R}
  pop = urine2plasma(pop)
  istwoplots = linear && loglinear
  istwoplots && (layout --> (1, 2))
  hastitle = length(get(plotattributes, :title, [])) >= 2
  hasunits = eltype(C) <: Quantity
  isurine  = R !== Nothing
  timename = isurine ? "Midpoint" : "Time"
  hasunits && (timename *= " ($(string(unit(eltype(T)))))")
  concname = isurine ? "Excretion Rate" : "Concentration"
  hasunits && (concname *= " ($(string(unit(eltype(C)))))")
  xguide --> timename
  yguide --> concname
  label --> [subj.id for subj in pop]
  linestyle --> :auto
  legend --> false
  timearr = [ustrip.(subj.abstime) for subj in pop]
  concarr = [replace!(x->x < eps(one(x)) ? eps(one(x)) : x, vcat(map(x->ustrip.(x), subj.conc)...)) for subj in pop]
  if linear
    @series begin
      seriestype --> :path
      istwoplots && (subplot --> 1)
      _title = hastitle ? plotattributes[:title][1] : "Linear view"
      title := _title
      (timearr, concarr)
    end
  end
  if loglinear
    @series begin
      concname = hasunits ? "Concentration ($(string(unit(eltype(C)))))" : "Concentration"
      yguide --> concname
      yscale --> :log10
      seriestype --> :path
      istwoplots && (subplot --> 2)
      _title = hastitle ? plotattributes[:title][2] : "Semilogrithmic view"
      title := _title
      (timearr, concarr)
    end
  end
end
