using Test
using PuMaS, Distributions

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(joinpath(dirname(pathof(PuMaS)), "..", "examples/oral1_1cpt_KAVCL_MD_data.txt"),
                    covariates,dvs)

m_diffeq = @model begin

    @data_cov ka cl v
    
    @collate begin
        Ka = ka
        CL = cl
        V  = v
    end
    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    @post cp = Central / V
    
    @error begin
        conc = Central / V
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

    @post cp = Central / V

    @error begin
        conc = Central / V
        dv ~ Normal(conc, 1e-100)
    end
end

@test_broken @inferred pkpd_solve(m_analytic,data[1],(),())
@test_broken @inferred pkpd_post(m_analytic,data[1],(),())
@test_broken @inferred pkpd_simulate(m_analytic,data[1],(),())

# inference broken in both `modify_pkpd_problem` and `solve`
@test_broken @inferred pkpd_solve(m_diffeq,data[1],(),())
@test_broken @inferred pkpd_post(m_analytic,data[1],(),())
@test_broken @inferred pkpd_simulate(m_diffeq,data[1],(),()) 
