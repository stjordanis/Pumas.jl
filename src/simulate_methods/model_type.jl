struct PKPDModel{P,SP,R}
    prob::P
    set_parameters::SP
    reduction::R
end

abstract type ErrorModel end

struct FullModel{P<:PKPDModel,E<:ErrorModel}
    pkpd::P
    err::E
end

### High level simulate functions for persons

function PKPDModel(prob,set_parameters=nothing)
    reduction=(sol,p,datai) -> (sol,false)
    PKPDModel{typeof(prob),typeof(set_parameters),typeof(reduction)}(
                                                prob,set_parameters,reduction)
end

function simulate(m::PKPDModel,θ,η,data::Person,
                  args...;kwargs...)

    simulate(m.prob,m.set_parameters,θ,η,data,
                m.reduction,
                args...;kwargs...)
end

function simulate(m::FullModel, θ, η, data::Person, args...; kwargs...)
    ui = simulate(m.pkpd, θ,η, data,args...; kwargs...)
    simulate(m.err, θ, ui)
end

#### Simulate populations

function generate_η(ω,N)
  [rand(MvNormal(zeros(size(ω,1)),ω)) for i in 1:N]
end

function simulate(m::Union{PKPDModel,FullModel},θ,ω,data::Population,args...;
                  parallel_type=:threads,kwargs...)
    N = length(data)
    η = generate_η(ω,N)
    t = @elapsed u = [simulate(m,θ,η[i],data[i],args...;kwargs...) for i in 1:N]
    MonteCarloSolution(u,t,true)
end
