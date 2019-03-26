using PuMaS.NCA, Test, CSV
using PuMaS
using Random

file = PuMaS.example_nmtran_data("nca_test_data/patient_data_test_sk")
df = CSV.read(file)
data = @test_nowarn parse_ncadata(df, id=:ID, time=:Time, conc=:Prefilter_Conc, warn=false, formulation=:Mode, iv=Inf, amt=:Amount, duration=:Infusion_Time)
@test data[1].dose === NCADose(0.0, 750, nothing, 0.25, NCA.IVInfusion)
