using PuMaS.NCA, Test, CSV
using PuMaS
using Random

file = PuMaS.example_nmtran_data("nca_test_data/patient_data_test_sk")
df = CSV.read(file)
data = @test_nowarn parse_ncadata(df, id=:ID, time=:Time, conc=:Prefilter_Conc, warn=false, formulation=:Mode, reference=Inf, amt=:Amount, duration=:Infusion_Time)
@test data[1].dose === NCADose(0.0, 750, 0.25, NCA.IVInfusion)
@test NCA.mrt(data; auctype=:last)[:mrt] == NCA.aumclast(data)[:aumclast]./NCA.auclast(data)[:auclast] .- 0.25/2
@test NCA.mrt(data; auctype=:inf)[:mrt] == NCA.aumc(data)[:aumc]./NCA.auc(data)[:auc] .- 0.25/2
