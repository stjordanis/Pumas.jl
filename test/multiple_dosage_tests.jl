using Base.Test
using PKPDSimulator, NamedTuples

# Load data
covariates = [:ka, :cl, :v]
dvs = [:dv]
data = process_data(Pkg.dir("PKPDSimulator", "examples/oral1_1cpt_KAVCL_MD_data.txt"),
                    covariates,dvs)

m = @model begin

    @data_cov ka cl v
    
    # TODO: this shouldn't be necessary
    @collate begin
        Ka = ka
        CL = cl
        V  = v
    end
    @dynamics begin
        dDepot   = -Ka*Depot
        dCentral =  Ka*Depot - (CL/V)*Central
    end

    # we approximate the error by computing the loglikelihood
    @error begin
        conc = Central / V
        dv ~ Normal(conc, 1.0)
    end

end

x0 = ()
y0 = ()
subject1 = data.subjects[1]
sol1 = pkpd_simulate(m,data.subjects[1],x0,y0,abstol=1e-14,reltol=1e-14)
sol2 = pkpd_simulate(m,data.subjects[2],x0,y0,abstol=1e-14,reltol=1e-14)

# TODO: check these actually match up with NONMEM
