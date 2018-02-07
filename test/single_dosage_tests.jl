using Base.Test
using PKPDSimulator, NamedTuples, Distributions, PDMats

# Read the data
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                    [:sex,:wt,:etn],separator=',')
# add a small epsilon to time 0 observations
for subject in data.subjects
    obs1 = subject.observations[1]
    if obs1.time == 0
        subject.observations[1] = PKPDSimulator.Observation(sqrt(eps()), obs1.val, obs1.cmt)
    end
end


m = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        σ ∈ RealDomain(lower=0.0, init=1.0)
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end

    @data_cov sex wt etn

    @collate begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @error begin
        conc = Central / V
        dv ~ Normal(conc, conc*σ)
    end
end

# Define the ODE
x0 = @NT(θ = [2.268,74.17,468.6,0.5876],
         ω = PDMat([0.05 0.0;
                    0.0 0.2]),
         σ = 0.1)

y0 = init_random(m, x0)

subject1 = data.subjects[1]
sol1 = pkpd_simulate(m,subject1,x0,y0)

pkpd_likelihood(m,subject1,x0,y0)
pkpd_simulate(m,subject1,x0,y0)
pkpd_simulate(m,subject1,x0)



