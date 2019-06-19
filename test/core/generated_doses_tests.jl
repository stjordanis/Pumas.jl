using Test, LinearAlgebra
using PuMaS

# Gut dosing model
m_diffeq = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
    end

    @random begin
        η ~ MvNormal(Matrix{Float64}(I, 3, 3))
    end

    @covariates isPM Wt

    @pre begin
        TVCL = isPM == "yes" ? θ[1] : θ[4]
        CL = θ[1]*(Wt/70)^0.75*exp(η[1])
        V = θ[2]*(Wt/70)^0.75*exp(η[2])
        Ka = θ[3]*exp(η[3])
    end

    @vars begin
        cp = Central/V
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @derived begin
        conc = @. Central / V
        dv ~ @. Normal(conc, 0.2)
    end
end

param = (
    θ = [0.4,20,1.1,2],
    σ_prop = 0.04
    )

randeffs = (η=randn(3),)

subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2),
                  cvs = cvs=(isPM="no", Wt=70))

# Make sure simobs works without time, defaults to 1 day, obs at each hour
obs = simobs(m_diffeq, subject, param, randeffs)
@test obs.times == 0.0:1.0:84.0
@test DataFrame(obs; include_events=false).time == 0.0:1.0:84.0
@test all(DataFrame(obs).isPM .== "no")

#=
using Plots
plot(obs)
=#

pop = Population([Subject(id = id,
            evs = DosageRegimen([10rand(), 20rand()],
            ii = 24, addl = 2, ss = 1:2, time = [0, 12],
            cmt = 2),cvs = cvs=(isPM="no", Wt=70)) for id in 1:10])
pop_obs1 = simobs(m_diffeq, pop, param, randeffs)
_data = DataFrame(pop_obs1)

#=
plot(pop_obs)
=#

data = read_pumas(_data,cvs = [:isPM,:Wt],dvs = [:conc,:dv])

obs2 = simobs(m_diffeq, data[1], param, randeffs)
@test_broken _data2 = DataFrame(obs2)

pop_obs2 = simobs(m_diffeq, data, param, randeffs)
@test_broken _data2 = DataFrame(pop_obs2)
#@test all(_data.conc .== _data2.conc)
#@test !all(_data.dv .== _data2.dv)
