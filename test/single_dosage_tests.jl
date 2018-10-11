using PuMaS, Test, CSV, Distributions, PDMats, Random, StaticArrays

# Read the data
data = process_nmtran(CSV.read(joinpath(joinpath(dirname(pathof(PuMaS)), ".."),"examples/data1.csv")),
                      [:sex,:wt,:etn])

# add a small epsilon to time 0 observations
for subject in data.subjects
    obs1 = subject.observations[1]
    if obs1.time == 0
        subject.observations[1] = PuMaS.Observation(sqrt(eps()), obs1.val, obs1.cmt)
    end
end

# Definition using diffeqs
m_diffeq = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        σ ∈ RealDomain(lower=0.0, init=1.0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn

    @collate begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @dynamics begin
        cp       =  Central/V
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @post begin
        conc = Central / V
        dv ~ Normal(conc, conc*σ)
    end
end

# Definition using analytic models
m_analytic = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        σ ∈ RealDomain(lower=0.0, init=1.0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn

    @collate begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @dynamics OneCompartmentModel

    @post begin
        conc = Central / V
        dv ~ Normal(conc, conc*σ)
    end
end

# Define the ODE
x0 = (θ = [2.268,74.17,468.6,0.5876],
      Ω = PDMat([0.05 0.0;
                 0.0 0.2]),
      σ = 0.1)

subject1 = data.subjects[1]

y0 = init_random(m_diffeq, x0)

sol_diffeq   = solve(m_diffeq,subject1,x0,y0)
sol_analytic = solve(m_analytic,subject1,x0,y0)

@test sol_diffeq(1.0) ≈ sol_analytic(1.0) rtol=1e-4

sol_diffeq   = solve(m_diffeq,subject1,x0,y0,Rosenbrock23())

@test sol_diffeq.alg == Rosenbrock23()

@test likelihood(m_diffeq,subject1,x0,y0) ≈ likelihood(m_analytic,subject1,x0,y0)

sim_diffeq = begin
    Random.seed!(1)
    s = simobs(m_diffeq,subject1,x0,y0)[:dv]
end
sim_analytic = begin
    Random.seed!(1)
    s = simobs(m_analytic,subject1,x0,y0)[:dv]
end
@test sim_diffeq ≈ sim_analytic rtol=1e-4

sim_diffeq = begin
    Random.seed!(1)
    s = simobs(m_diffeq,subject1,x0)[:dv]
end
sim_analytic = begin
    Random.seed!(1)
    s = simobs(m_analytic,subject1,x0)[:dv]
end
@test sim_diffeq ≈ sim_analytic rtol=1e-4

sol_diffeq = solve(m_diffeq,data,x0, parallel_type = PuMaS.Serial)
sol_diffeq = solve(m_diffeq,data,x0, parallel_type = PuMaS.Threading)
sol_diffeq = solve(m_diffeq,data,x0, parallel_type = PuMaS.Distributed)
@test_broken sol_diffeq = solve(m_diffeq,data,x0, parallel_type = PuMaS.SplitThreads)
