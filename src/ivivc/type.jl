# IVIVCSubject
mutable struct VivoSubject{ID, C, T, F, D, pType} <: Ivivc
  id::ID
  conc::C
  time::T
  form::F
  dose::D
  # params which are needed for modeling
  m::Function                           # model type
  alg::Optim.FirstOrderOptimizer        # alg to optimize cost function
  p0::pType                             # intial values of params
  ub::Union{Nothing, pType}             # upper bound of params
  lb::Union{Nothing, pType}             # lower bound of params
  pmin::pType                           # optimized params
  function VivoSubject(conc, time, form, dose, id=1)
    return new{typeof(id), typeof(conc), typeof(time),
                typeof(form), typeof(dose), typeof(conc)}(id, conc, time, form, dose)
  end
end

Base.Broadcast.broadcastable(q::VivoSubject) = Ref(q)
function Base.show(io::IO, n::VivoSubject)
  println(io, "VivoSubject:")
  df = DataFrame(id = n.id, time = n.time, conc = n.conc, form = n.form, dose = n.dose)
  show(io::IO, df)
  # println(io, "  id:          $(n.id)")
  # println(io, "  conc:         $(n.conc)")
  # println(io, "  time:         $(n.time)")
  # println(io, "  formulation:  $(n.form)")
  # println(io, "  dose:         $(n.dose)")
end

# VivoPopulation
struct VivoPopulation{popType} <: Ivivc
  subjects::popType
  function VivoPopulation(_pop)
    # TODO: can do some checking and other stuff
    return new{typeof(_pop)}(_pop)
  end
end

# VitroSubject
mutable struct VitroSubject{ID, C, T, F, pType} <: Ivivc
  id::ID
  conc::C
  time::T
  form::F
  # params which are needed for modeling
  m::Function                           # model type
  alg::Optim.FirstOrderOptimizer        # alg to optimize cost function
  p0::pType                             # intial values of params
  ub::pType                             # upper bound of params
  lb::pType                             # lower bound of params
  pmin::pType                           # optimized params
  function VitroSubject(conc, time, form, id=1)
    return new{typeof(id), typeof(conc), typeof(time),
                typeof(form), typeof(conc)}(id, conc, time, form)
  end
end

Base.Broadcast.broadcastable(q::VitroSubject) = Ref(q)
function Base.show(io::IO, n::VitroSubject)
  println(io, "VitroSubject:")
  df = DataFrame(id = n.id, time = n.time, conc = n.conc, form = n.form)
  show(io::IO, df)
  # println(io, "  id:          $(n.id)")
  # println(io, "  conc:         $(n.conc)")
  # println(io, "  time:         $(n.time)")
  # println(io, "  formulation:  $(n.form)")
end

# VitroPopulation
struct VitroPopulation{popType} <: Ivivc
  subjects::popType
  function VitroPopulation(_pop)
    # TODO: can do some checking and other stuff
    return new{typeof(_pop)}(_pop)
  end
end
