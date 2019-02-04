using Test, LinearAlgebra
using PuMaS

# Gut dosing model
m_diffeq = @model begin
    @param begin
        θ ∈ VectorDomain(3, lower=zeros(3), init=ones(3))
    end

    @random begin
        η ~ MvNormal(Matrix{Float64}(I, 2, 2))
    end

    @pre begin
        Ka = θ[1]
        CL = θ[2]*exp(η[1])
        V  = θ[3]*exp(η[2])
    end

    @dynamics begin
        cp       =  Central/V
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @derived begin
        conc = @. Central / V
        dv ~ @. Normal(conc, 0.2)
    end
end

x0 = (θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0  #V
     ],)
y0 = init_random(m_diffeq, x0)

subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))

# Make sure simobs works without time, defaults to 1 day, obs at each hour
obs = simobs(m_diffeq, subject, x0, y0)
@test obs.times == 0.0:1.0:24.0
@test DataFrame(obs)[:,1] == 0.0:1.0:24.0

using Plots
plot(obs)
