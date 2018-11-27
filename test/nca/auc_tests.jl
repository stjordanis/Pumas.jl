using PuMaS.NCA, Test, CSV
using PuMaS
using Random

root = joinpath(dirname(pathof(PuMaS)), "..")
data = CSV.read("$root/examples/nca_test_data/dapa_IV.csv")

conc = Float64.(data[:CObs])
t = Float64.(data[:TIME])

data = CSV.read("$root/examples/nca_test_data/Final_Parameters_Pivoted.csv")

correct_auc = Float64.(data[:AUCINF_obs])
correct_auc_last = Float64.(data[:AUClast])
correct_aumc = Float64.(data[:AUMCINF_obs])

test_auc = rand(24,)
test_aumc = rand(24,)

idx = 1:16

@test ctlast(zeros(5), 1:5) === (clast = missing, tlast = missing)
@test ctlast(1:5, 1:5) === (clast = 5, tlast = 5)
@test ctlast(conc[idx], t[idx]) === (clast = conc[idx][end], tlast = t[idx][end])
arr = [missing, 1, 2, 3, missing]
@test ctlast(arr, 1:5) === (clast = 3, tlast = 4)
@test ctmax(arr, 1:5) === (cmax = 3, tmax = 4)
@test ctmax(conc[idx], t[idx]) === (cmax = conc[idx][1], tmax = t[idx][1])
@test ctmax(conc[idx], t[idx], interval=(2,Inf)) === (cmax = conc[idx][7], tmax = t[idx][7])
@test ctmax(conc[idx], t[idx], interval=(24.,Inf)) === (cmax = conc[idx][end], tmax = t[idx][end])
@test_throws ArgumentError ctmax(conc[idx], t[idx], interval=(100,Inf))

for m in (:linear, :linuplogdown)
  @inferred auc(conc[idx], t[idx], method=m)
  @inferred aumc(conc[idx], t[idx], method=m)
  @test_nowarn NCA.interpextrapconc(conc[idx], t[idx], 1000rand(500), interpmethod=:linear)
  @test_nowarn auc(conc[idx], t[idx], method=m, interval=(0,100.))
  @test_nowarn aumc(conc[idx], t[idx], method=m, interval=(0,100.))
end

@test find_lambdaz(conc[idx], t[idx], idxs=12:16)[1] == find_lambdaz(conc[idx], t[idx])[1]
@test find_lambdaz(conc[idx], t[idx], idxs=12:16)[1] ≈ data[:Lambda_z][1] atol=1e-6

fails = (6, 9)
for i in 1:24
  idx = 16(i-1)+1:16*i
  aucs = auc(conc[idx], t[idx], method=:linear)
  aumcs = aumc(conc[idx], t[idx], method=:linear)
  @test_nowarn auc(conc[idx], t[idx], method=:linuplogdown)
  @test_nowarn aumc(conc[idx], t[idx], method=:linuplogdown)
  @test data[:AUClast][i] ≈ aucs[1] atol = 1e-6
  @test data[:AUMClast][i] ≈ aumcs[1] atol = 1e-6
  if i in fails
    @test_broken data[:AUCINF_obs][i] ≈ aucs[2] atol = 1e-6
    @test_broken data[:AUMCINF_obs][i] ≈ aumcs[2] atol = 1e-6
    @test_broken data[:Lambda_z][i] ≈ aucs[3] atol = 1e-6
    @test_broken data[:Lambda_z][i] ≈ aumcs[3] atol = 1e-6
  else
    @test data[:AUCINF_obs][i] ≈ aucs[2] atol = 1e-6
    @test data[:AUMCINF_obs][i] ≈ aumcs[2] atol = 1e-6
    @test data[:Lambda_z][i] ≈ aucs[3] atol = 1e-6
    @test data[:Lambda_z][i] ≈ aumcs[3] atol = 1e-6
  end
end
