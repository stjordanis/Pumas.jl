function immediate_absorption_f(t,t0,u0,dose,p,aux)
  D = dose+u0
  next_aux = exp(-(t-t0)*p.Ka)*D
  next_u,nothing
end

function ImmediateAbsorptionModel(tf)
  AnalyticalProblem(immediate_absorption_f,0.0,nothing,(0.0,tf))
end

export ImmediateAbsorptionModel

function one_compartment_f(t,t0,u0,dose,p,aux)
  D = dose+aux
  next_aux = exp(-(t-t0)*p.Ka)*D
  next_u = exp(-t*(p.Ka+p.CL/p.V))*(D*exp(p.Ka*t0+p.CL*t/p.V)*p.Ka*p.V +
  exp(t*p.Ka+p.CL*t0/p.V)*(u0*p.CL - (u0 + D)*p.Ka*p.V))/(p.CL-p.Ka*p.V)
  next_u,next_aux
end

function OneCompartmentModel(tf)
  AnalyticalProblem(one_compartment_f,0.0,0.0,(0.0,tf))
end

export OneCompartmentModel
