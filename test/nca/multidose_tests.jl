using PuMaS.NCA, Test, CSV
using PuMaS

multiple_doses_file = PuMaS.example_nmtran_data("nca_test_data/dapa_IV_ORAL")
mdata = CSV.read(multiple_doses_file)

mncapop = @test_nowarn parse_ncadata(mdata, time=:TIME, conc=:COBS, amt=:AMT, formulation=:FORMULATION, occasion=:OCC)

@test_nowarn bioavailability(mncapop)
