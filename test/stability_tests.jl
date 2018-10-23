using PuMaS, Test, Distributions

# Load data
cvs = [:ka, :cl, :v]
dvs = [:dv]
data = process_nmtran(joinpath(dirname(pathof(PuMaS)), "..", "examples/oral1_1cpt_KAVCL_MD_data.csv"),
                      cvs, dvs)

m_diffeq = @model begin

    @covariates ka cl v

    @pre begin
        Ka = ka
        CL = cl
        V  = v
    end
    @dynamics begin
        cp       =  Central/V
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @post begin
        conc = Central / V
        dv ~ Normal(conc, 1e-100)
    end
end

m_analytic = @model begin

    @covariates ka cl v

    @pre begin
        Ka = ka
        CL = cl
        V  = v
    end
    @dynamics OneCompartmentModel

    @post begin
        conc = Central / V
        dv ~ Normal(conc, 1e-100)
    end
end

@test_broken @inferred solve(m_analytic,data[1],(),())
@test_broken @inferred simobs(m_analytic,data[1],(),())
@test_broken @inferred simobs(m_analytic,data[1],(),())

# inference broken in both `modify_pkpd_problem` and `solve`
@test_broken @inferred solve(m_diffeq,data[1],(),())
@test_broken @inferred simobs(m_analytic,data[1],(),())
@test_broken @inferred simobs(m_diffeq,data[1],(),())
