using PuMaS.NCA, Test, CSV
using PuMaS
using Random

file = PuMaS.example_nmtran_data("nca_test_data/masked_sort_data")
df = CSV.read(file, missingstring="NA")
data1 = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, occasion=:PERIOD, group=:METABOLITE, warn=false, ii=0.1)
data2 = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, occasion=:PERIOD, group=[:METABOLITE], warn=false)
for data in [data1, data2]
  io = IOBuffer()
  show(io, "text/plain", data)
  @test occursin("5 subjects", String(take!(io)))
  @test map(s->s.group, data) == repeat(["METABOLITE" => "Metabolite $i" for i in 1:4], inner=5)
  @test_nowarn display(data)
  @test_nowarn display(data[1])
  aucs = @test_nowarn NCA.auc(data)
  @test names(aucs) == [:id, :occasion, :METABOLITE, :auc]
  @test aucs[:METABOLITE] == repeat(["Metabolite $i" for i in 1:4], inner=5*3)
  @test_nowarn NCA.mrt(data)
  @test_nowarn NCA.tmin(data)
end
data3 = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, group=[:PERIOD, :METABOLITE], warn=false) # multi-group
aucs = @test_nowarn NCA.auc(data3)
@test aucs[:METABOLITE][1] == "Metabolite 1"
@test aucs[:PERIOD] == repeat(["$i" for i in 1:3], inner=20)

# test ii
ii1 = [0.1 * i for i in eachindex(data1)]
pop = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, occasion=:PERIOD, group=[:METABOLITE], warn=false, ii=ii1)
@test all(i -> pop[i].ii == ii1[i], eachindex(pop))
ii2 = [0.1 * i for i in 1:5]
pop = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, occasion=:PERIOD, group=[:METABOLITE], warn=false, ii=ii2)
fullii2 = repeat(ii2, 5)
@test all(i -> pop[i].ii == fullii2[i], eachindex(pop))
add_ii!(pop, 0.0) # test add_ii
add_ii!(pop, ii2)
@test all(i -> pop[i].ii == fullii2[i], eachindex(pop))
# length error
ii3 = [0.1 * i for i in 1:6]
@test_throws ArgumentError read_nca(df, id=:ID, time=:TIME, conc=:TCONC, occasion=:PERIOD, group=[:METABOLITE], warn=false, ii=ii3)
