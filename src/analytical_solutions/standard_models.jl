function immediate_absorption_f(t,t0,u0,dose,p,aux)
  D = dose+u0
  next_aux = exp(-(t-t0)*p.Ka)*D
  next_u,nothing
end

function ImmediateAbsorptionModel(tf)
  AnalyticalProblem(immediate_absorption_f,0.0,nothing,(0.0,tf))
end

export ImmediateAbsorptionModel

function one_compartment_f(t,t0,C0,dose,p,D0)
  Ka = p.Ka              # absorption rate
  Ke = p.CL / p.V        # elimination rate
  D0 += dose             # initial depot value
  Sa = exp(-(t-t0)*Ka)
  Se = exp(-(t-t0)*Ke)
  D  = D0 * Sa           # next depot
  C =  Ka / (Ka - Ke) * D0 * (Se - Sa) + C0 * Se # next central
  C,D
end

function OneCompartmentModel(tf)
  AnalyticalProblem(one_compartment_f,0.0,0.0,(0.0,tf))
end

export OneCompartmentModel
