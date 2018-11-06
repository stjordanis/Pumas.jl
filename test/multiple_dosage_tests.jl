using PuMaS, Test, Random

# Load data
cvs = [:ka, :cl, :v]
dvs = [:dv]
data = process_nmtran(example_nmtran_data("oral1_1cpt_KAVCL_MD_data"),
                      cvs, dvs)

m_diffeq = @model begin

    @covariates ka cl v

    @dynamics begin
        cp       =  cl/v
        Depot'   = -ka*Depot
        Central' =  ka*Depot - cp*Central
    end

    # we approximate the error by computing the logconditional_loglikelihood
    @derived begin
        conc = @. Central / v
        dv ~ @. Normal(conc, 1e-100)
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

    # we approximate the error by computing the logconditional_loglikelihood
    @derived begin
        conc = @. Central / v
        dv ~ @. Normal(conc, 1e-100)
    end
end

subject1 = data.subjects[1]
x0 = ()
y0 = ()

sol_diffeq   = solve(m_diffeq,subject1,x0,y0)
sol_analytic = solve(m_analytic,subject1,x0,y0)

@test sol_diffeq(95.99) ≈ sol_analytic(95.99) rtol=1e-4
@test sol_diffeq(217.0) ≈ sol_analytic(217.0) rtol=1e-3 # TODO: why is this so large?

sim_diffeq = begin
    Random.seed!(1)
    s = simobs(m_diffeq,subject1,x0,y0)[:dv]
end
sim_analytic = begin
    Random.seed!(1)
    s = simobs(m_analytic,subject1,x0,y0)[:dv]
end
@test sim_diffeq ≈ sim_analytic rtol=1e-3
