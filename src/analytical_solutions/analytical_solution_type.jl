struct PKPDAnalyticalSolution{T,N,uType,tType,dType,rType,pType,P} <: DiffEqBase.AbstractAnalyticalSolution{T,N}
  u::uType
  t::tType
  doses::dType
  rates::rType
  p::pType
  prob::P
  dense::Bool
  tslocation::Int
  retcode::Symbol
  continuity::Symbol
end

function (sol::PKPDAnalyticalSolution)(t,deriv::Type{<:Val}=Val{0};idxs=nothing,continuity=sol.continuity)
  i = searchsortedfirst(sol.t,t) - 1
  if i < length(sol.t) && t == sol.t[i+1] && continuity == :right
    # If at a dose time and using right continuity, return the saved value
    # with dose appled.
    res = sol.u[i+1] + sol.doses[i+1]
  elseif i == 0
    # If before any saved time points, return the initial value.
    res = sol.prob.u0
  else
    # In all other cases, propagate from the last saved time point.
    t0 = sol.t[i]
    u0 = sol.u[i]
    dose = sol.doses[i]
    rate = sol.rates[i]
    res = sol.prob.f(t,t0,u0,dose,sol.p,rate)
  end
  idxs==nothing ? res : res[idxs]
end

function (sol::PKPDAnalyticalSolution)(ts::AbstractArray,deriv::Type{<:Val}=Val{0};idxs=nothing,continuity=sol.continuity)
  if idxs != nothing
    u = Vector{eltype(sol.u[1][idxs])}(undef, length(ts))
  else
    u = Vector{eltype(sol.u)}(undef, length(ts))
  end
  for j in 1:length(ts)
    t = ts[j]
    u[j] = sol(t,deriv;idxs=idxs,continuity=continuity)
  end
  DiffEqBase.DiffEqArray(u,ts)
end

Base.summary(A::PKPDAnalyticalSolution) = string("Timeseries Solution with uType ",eltype(A.u)," and tType ",eltype(A.t))
function Base.show(io::IO, A::PKPDAnalyticalSolution)
  println(io,string("retcode: ",A.retcode))
  print(io,"t: ")
  show(io, A.t)
  println(io)
  print(io,"u: ")
  show(io, A.u)
end
function Base.show(io::IO, m::MIME"text/plain", A::PKPDAnalyticalSolution)
  println(io,string("retcode: ",A.retcode))
  print(io,"t: ")
  show(io,m,A.t)
  println(io)
  print(io,"u: ")
  show(io,m,A.u)
end

struct AnalyticalPKSolution{T,N,uType,tType,S1,S2} <: DiffEqBase.AbstractTimeseriesSolution{T,N}
  u::Vector{uType}
  t::Vector{tType}
  pksol::S1
  numsol::S2
  retcode::Symbol
  AnalyticalPKSolution(u,t,pksol,numsol) = AnalyticalPKSolution(u,t,pksol,numsol,(size(u[1])..., length(u)))
  AnalyticalPKSolution(u::AbstractVector{T},t,pksol,numsol,dims::NTuple{N}) where {T, N} =
        new{eltype(T),N,eltype(u),eltype(t),typeof(pksol),typeof(numsol)}(u,t,pksol,numsol,
            pksol==:Success && numsol==:Success ? :Success : :Failure)
end
Base.summary(A::AnalyticalPKSolution) = string("Timeseries Solution with uType Float64 and tType Float64")
function Base.show(io::IO, A::AnalyticalPKSolution)
  println(io,string("retcode: ",A.retcode))
  print(io,"t: ")
  show(io, A.t)
  println(io)
  print(io,"u: ")
  show(io, A.u)
end
function Base.show(io::IO, m::MIME"text/plain", A::AnalyticalPKSolution)
  println(io,string("retcode: ",A.retcode))
  print(io,"t: ")
  show(io,m,A.t)
  println(io)
  print(io,"u: ")
  show(io,m,A.u)
end

function (sol::AnalyticalPKSolution)(t,deriv::Type{<:Val}=Val{0};idxs=nothing,continuity=sol.pksol.continuity)
  if idxs === nothing
    [sol.pksol(t,deriv;idxs=idxs,continuity=continuity);sol.numsol(t,deriv;idxs=idxs,continuity=continuity)]
  elseif idxs isa Number
    pksize = length(sol.pksol.u[1])
    if idxs <= pksize
      sol.pksol(t,deriv;idxs=idxs,continuity=continuity)
    else
      sol.numsol(t,deriv;idxs=idxs-pksize,continuity=continuity)
    end
  else
    error("Non-number idxs not supported with mixed analytical + numerical yet. Please file an issue.")
  end
end

# No extra specialization yet
# Can make the DiffEqArray more directly...
function (sol::AnalyticalPKSolution)(ts::AbstractArray,deriv::Type{<:Val}=Val{0};idxs=nothing,continuity=sol.pksol.continuity)
  if idxs === nothing
    pku = sol.pksol(ts,deriv;idxs=idxs,continuity=continuity)
    nsu = sol.numsol(ts,deriv;idxs=idxs,continuity=continuity)
    u = [[p;n] for (p,n) in zip(pku.u,nsu.u)]
    return DiffEqArray(u,ts)
  elseif idxs isa Number
    pksize = length(sol.pksol.u[1])
    if idxs <= pksize
      return sol.pksol(ts,deriv;idxs=idxs,continuity=continuity)
    else
      return sol.numsol(ts,deriv;idxs=idxs-pksize,continuity=continuity)
    end
  else
    error("Non-number idxs not supported with mixed analytical + numerical yet. Please file an issue.")
  end
end
