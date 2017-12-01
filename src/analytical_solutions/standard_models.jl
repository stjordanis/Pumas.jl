function immediate_absorption_f(t,t0,C0,dose,p,rate)
    Ke = p.CL/p.V
    C0 += dose
    rKe = rate/Ke
    (rKe + exp(-(t-t0)*Ke) * (-rKe + C0))
end

function ImmediateAbsorptionModel(tf)
  AnalyticalProblem(immediate_absorption_f,0.0,(0.0,tf))
end

export ImmediateAbsorptionModel

function one_compartment_f(t,t0,amounts,doses,p,rates)
  Ka = p.Ka
  Ke = p.CL/p.V         # elimination rate
  amt = amounts + doses             # initial values for cmt's + new doses
  Sa = exp(-(t-t0)*Ka)
  Se = exp(-(t-t0)*Ke)
  Depot  = (amt[1] * Sa) + rates[1]/(Ka*(1-Sa))          # next depot (cmt==1)
  Central =  Ka / (Ka - Ke) * (amt[1] * (Se - Sa) + rates[1]*((1-Se)/Ke - (1-Sa)/Ka)) +
            amt[2] * Se + rates[2]/Ke*(1-Se) # next central (cmt==2)
  @SVector [Depot,Central]
end

#=
function one_compartment_f(t,t0,u0,dose,p,rate)
  D0,C0 = u0 + dose
  Ka = p.Ka              # absorption rate
  Ke = p.CL / p.V        # elimination rate
  Sa = exp(-(t-t0)*Ka)
  Se = exp(-(t-t0)*Ke)
  D  = D0 * Sa           # next depot
  C =  Ka / (Ka - Ke) * D0 * (Se - Sa) + C0 * Se # next central
  @SVector [D,C]
end
=#

function OneCompartmentModel(tf)
  AnalyticalProblem(one_compartment_f,@SVector([0.0,0.0]),(0.0,tf))
end

export OneCompartmentModel
