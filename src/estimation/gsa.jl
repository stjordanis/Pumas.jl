using DiffEqSensitivity

abstract type GSAMethod end
struct Sobol <: GSAMethod end
struct Morris <: GSAMethod end


function gsa(m::PumasModel,data::Population,params::NamedTuple,method::Sobol,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],N=1000,order=[0],args...; kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,data,param)
        collect(Iterators.flatten([sim.sims[i].observed.dv for i in 1:length(sim.sims)]))
    end
    DiffEqSensitivity.sobol_sensitivity(f,p_range,N,order,args...; kwargs...)
end

function gsa(m::PumasModel,data::Population,params::NamedTuple,method::Morris,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],
                p_steps=[100 for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],args...; kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,data,param)
        collect(Iterators.flatten([sim.sims[i].observed.dv for i in 1:length(sim.sims)]))
    end
    DiffEqSensitivity.morris_sensitivity(f,p_range,p_steps,args...; kwargs...)
end