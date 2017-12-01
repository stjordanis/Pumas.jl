struct PKPDAnalyticalSolution{T,N,uType,tType,dType,rType,pType,P} <: AbstractAnalyticalSolution{T,N}
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

function (sol::PKPDAnalyticalSolution)(t,deriv::Type=Val{0};idxs=nothing)
    i = searchsortedfirst(sol.t,t) - 1
    if i == 0
        return idxs==nothing ? sol.prob.u0 : sol.prob.u0[idxs]
    else
        t0 = sol.t[i].time
        u0 = sol.u[i]
        dose = sol.doses[i]
        rate = sol.rates[i]
        res = sol.prob.f(t,t0,u0,dose,sol.p,rate)
        return idxs==nothing ? res : res[idxs]
    end
end

function (sol::PKPDAnalyticalSolution)(ts::AbstractArray,deriv::Type=Val{0};idxs=nothing)
    if idxs != nothing
        u = Vector{eltype(sol.u[1][idxs])}(length(ts))
    else
        u = Vector{eltype(sol.u)}(length(ts))
    end
    for j in 1:length(ts)
        t = ts[j]
        i = searchsortedfirst(sol.t,t) - 1
        if i == 0
            res = sol.prob.u0
            u[j] = idxs==nothing ? res : res[idxs]
        else
            t0 = sol.t[i].time
            u0 = sol.u[i]
            dose = sol.doses[i]
            rate = sol.rates[i]
            res = sol.prob.f(t,t0,u0,dose,sol.p,rate)
            u[j] = idxs==nothing ? res : res[idxs]
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
