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
end

function (sol::PKPDAnalyticalSolution)(t,deriv::Type=Val{0};idxs=nothing,continuity=:right)
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

function (sol::PKPDAnalyticalSolution)(ts::AbstractArray,deriv::Type=Val{0};idxs=nothing,continuity=:right)
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
