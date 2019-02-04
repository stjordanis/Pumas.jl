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
        dv ~ @. Normal(conc, 1e-100)
    end
end

###############################
# Test 15
###############################


x0 = (θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0  #V
     ],)
y0 = init_random(m_diffeq, x0)

subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
sol = solve(m_diffeq, subject, x0, y0; tspan=(0.0,12.0+1e-14), abstol=1e-14, reltol=1e-14)
col = pre(m_diffeq, subject, x0, y0)
@test [1000 * sol(12*i + 1e-14)[2] / col.V for i in 0:1] ≈ [605.3220736386598;1616.4036675452326] atol=1e-8

# Make sure simobs works without time
obs = simobs(m_diffeq, subject, x0, y0)
@test obs.times == 0.0:1.0:24.0

###############################
# Test 16
###############################

subject = Subject(evs = DosageRegimen([10, 20, 10], ii = 24, ss = [1,2,1], time = 0:12:24, cmt = 2))
col = pre(m_diffeq, subject, x0, y0)
sol = solve(m_diffeq, subject, x0, y0; tspan=(0.0,60.0+1e-14), abstol=1e-14,reltol=1e-14)

@test [1000 * sol(12*i + 1e-14)[2] / col.V for i in 0:5] ≈ [605.3220736386598
                                                        1616.4036675452326
                                                        605.3220736387212
                                                        405.75952026789673
                                                        271.98874030537564
                                                        182.31950492267478] atol=1e-9
