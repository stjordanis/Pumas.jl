using Pumas.NCA, Test, CSV
using Pumas

file = Pumas.example_nmtran_data("nca_test_data/masked_sort_data")
df = CSV.read(file, missingstring="NA")
df[!,:ii] .= 12
df[!,:amt] .= 0.0
df.amt[1:23:end] .= 0.1
df[!,:route] .= "ev"
data = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, occasion=:PERIOD, group=[:METABOLITE], verbose=false)
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
data3 = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, group=[:PERIOD, :METABOLITE], verbose=false) # multi-group
aucs = @test_nowarn NCA.auc(data3)
@test aucs[:METABOLITE][1] == "Metabolite 1"
@test aucs[:PERIOD] == repeat(["$i" for i in 1:3], inner=20)

# test ii
df[!,:ii] .= [0.1 * i for i in df.ID]
pop = @test_nowarn read_nca(df, id=:ID, time=:TIME, conc=:TCONC, group=[:PERIOD, :METABOLITE], verbose=false)
