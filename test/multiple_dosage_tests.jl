using Test
using PuMaS, NamedTuples, Distributions

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(Pkg.dir("PuMaS", "examples/oral1_1cpt_KAVCL_MD_data.txt"),
                    covariates,dvs)

m_diffeq = @model begin

    @data_cov ka cl v
    
    @dynamics begin
        dDepot   = -ka*Depot
        dCentral =  ka*Depot - (cl/v)*Central
    end

    # we approximate the error by computing the loglikelihood
    @error begin
        conc = Central / v
        dv ~ Normal(conc, 1e-100)
    end
end

m_analytic = @model begin

    @data_cov ka cl v
    
    @collate begin
        Ka = ka
        CL = cl
        V  = v
    end
    @dynamics OneCompartmentModel

    # we approximate the error by computing the loglikelihood
    @error begin
        conc = Central / V
        dv ~ Normal(conc, 1e-100)
    end
end

subject1 = data.subjects[1]
x0 = ()
y0 = ()

sol_diffeq, _   = pkpd_solve(m_diffeq,subject1,x0,y0)
sol_analytic, _ = pkpd_solve(m_analytic,subject1,x0,y0)

@test sol_diffeq(95.99) ≈ sol_analytic(95.99) rtol=1e-4
@test sol_diffeq(217.0) ≈ sol_analytic(217.0) rtol=1e-3 # TODO: why is this so large?

sim_diffeq = begin
    srand(1)
    s = pkpd_simulate(m_diffeq,subject1,x0,y0)
    map(x-> x.dv, s)
end
sim_analytic = begin
    srand(1)
    s = pkpd_simulate(m_analytic,subject1,x0,y0)
    map(x-> x.dv, s)
end
@test sim_diffeq ≈ sim_analytic rtol=1e-3
