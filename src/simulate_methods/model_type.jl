struct PKPDModel{P,SP,R}
    prob::P
    set_parameters::SP
    reduction::R
end



abstract type ErrorModel end

"""
    Independent(errfn) <: ErrorModel

Assumes the error terms are independent.

`errfn` should be function which takes 2 arguments:
 - `uij`: the value at a particular timepoint of the reduced simulation
 - `θ`  : the population-level parameter vector

Example: proportional error:
```julia
properr(uij, θ) = Normal(uij, uij*θ[end])
```
"""
struct Independent{F} <: ErrorModel
    inderr::F
end


struct FullModel{P<:PKPDModel,E<:ErrorModel}
    pkpd::P
    err::E
end



function PKPDModel(prob,set_parameters=nothing,reduction=(sol,p,datai) -> (sol,false))
    PKPDModel{typeof(prob),typeof(set_parameters),typeof(reduction)}(
                                                prob,set_parameters,reduction)
end

export PKPDModel

function simulate(m::PKPDModel,θ,η,data::Person,
                  args...;kwargs...)

    simulate(m.prob,m.set_parameters,θ,η,data,
                m.reduction,
                args...;kwargs...)
end

function simulate(m::Independent, θ, ui)
    [rand(m.inderr(uij,θ)) for uij in ui]
end

function simulate(m::FullModel, θ, η, data::Person, args...; kwargs...)
    ui = simulate(m.pkpd, θ,η, data,args...; kwargs...)
    simulate(m.err, θ, ui)
end

