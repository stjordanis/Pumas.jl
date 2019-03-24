using PuMaS.NCA, Test, CSV
using PuMaS
using Random

file = PuMaS.example_nmtran_data("nca_test_data/masked_sort_data")
df = CSV.read(file, missingstring="NA")
groupby(df, [:METABOLITE])
parse_ncadata(df, id=:ID, time=:TIME, conc=:TCONC, occasion=:PERIOD, group=:METABOLITE)
