using PKPDSimulator, NamedTuples, Distributions

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:v,:cl,:ka]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/oral1_1cpt_KAVCL_SD_data.csv"),
                 covariates,dvs)
