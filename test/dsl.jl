using Base.Test

using PKPDSimulator, Distributions


# Read the data
covariates = [:sex,:wt,:etn]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,separator=',')

## parameters
m = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        Σ ∈ RealDomain(lower=0.0, init=1.0)
        a ∈ ConstDomain(0.2)
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
        dv ~ Normal(conc, conc*Σ)
    end
end

x0 = init(m.param)
y0 = map(mean, m.random(x0))

subject = data.subjects[1]

pkpd_simulate(m,subject,x0,y0)
pkpd_likelihood(m,subject,x0,y0)
