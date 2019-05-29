struct PKPDAnalyticalProblem{uType,tType,isinplace,F,S,C} <: DiffEqBase.AbstractAnalyticalProblem{uType,tType,isinplace}
  f::F
  u0::uType
  tspan::Tuple{tType,tType}
  ss::S
  callback::C
  function PKPDAnalyticalProblem{iip}(f,u0,tspan,ss = nothing,callback = nothing) where {iip}
    new{typeof(u0),promote_type(map(typeof,tspan)...),iip,
    typeof(f),typeof(ss),typeof(callback)}(f,u0,tspan,ss,callback)
  end
end

function PKPDAnalyticalProblem(f,u0,tspan,args...;kwargs...)
  iip = DiffEqBase.isinplace(f,7)
  PKPDAnalyticalProblem{iip}(f,u0,tspan,args...;kwargs...)
end

struct AnalyticalPKProblem{P1<:ExplicitModel,P2}
  pkprob::P1
  prob2::P2
end

export PKPDAnalyticalProblem, AnalyticalPKProblem
