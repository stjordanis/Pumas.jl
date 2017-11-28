struct PKPDModel{P,SP,R}
    prob::P
    set_parameters::SP
    reduction::R
end

function PKPDModel(prob,set_parameters=nothing,reduction=(sol,p,datai) -> (sol,false))
    PKPDModel{typeof(prob),typeof(set_parameters),typeof(reduction)}(
                                                prob,set_parameters,reduction)
end

export PKPDModel

function simulate(m::PKPDModel,θ,η,data,
                  ϵ = nothing, error_model = nothing,
                  args...;kwargs...)

    simulate(m.prob,m.set_parameters,θ,η,data,
                output_reduction = m.reduction,
                ϵ = nothing, error_model = nothing,args...;kwargs...)
end
