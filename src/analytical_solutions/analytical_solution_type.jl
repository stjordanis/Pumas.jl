struct PKPDAnalyticalSolution{T,N,uType,AType,tType,pType,P,E} <: AbstractAnalyticalSolution{T,N}
    u::uType
    aux::AType
    t::tType
    p::pType
    prob::P
    events::E
    dense::Bool
    tslocation::Int
    retcode::Symbol
end

function (sol::PKPDAnalyticalSolution)(t,deriv::Type=Val{0};idxs=nothing)
    i = searchsortedfirst(sol.t,t) - 1
    if i == 0
        return sol.prob.u0
    else
        t0 = sol.t[i]
        u0 = sol.u[i]
        return first(sol.prob.f(t,t0,u0,sol.events[i].amt,sol.p,sol.aux[i]))
    end
end

function (sol::PKPDAnalyticalSolution)(ts::AbstractArray,deriv::Type=Val{0};idxs=nothing)
    u = Vector{eltype(sol.u)}(length(ts))
    for j in 1:length(ts)
        t = ts[j]
        i = searchsortedfirst(sol.t,t) - 1
        if i == 0
            u[j] = sol.prob.u0
        else
            t0 = sol.t[i]
            u0 = sol.u[i]
            _u,_ = sol.prob.f(t,t0,u0,sol.events[i].amt,sol.p,sol.aux[i])
            u[j] = _u
        end
    end
    u
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
