using PuMaS.NCA, Test, CSV
using PuMaS

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

ctlast(conc[idx], t[idx]) === (clast = conc[idx][end], tlast = t[idx][end])
ctmax(conc[idx], t[idx]) === (cmax = conc[idx][1], tmax = t[idx][1])
ctmax(conc[idx], t[idx], interval=(2,Inf)) === (cmax = conc[idx][7], tmax = t[idx][7])

for m in (:linear, :log_linear)
  @inferred auc(conc[idx], t[idx], method=m)
  @inferred aumc(conc[idx], t[idx], method=m)
end

@test find_lambdaz(conc[idx], t[idx], idxs=12:16) == find_lambdaz(conc[idx], t[idx])
@test find_lambdaz(conc[idx], t[idx], idxs=12:16) ≈ data[:Lambda_z][1] atol=1e-6

fails = (6, 9)
for i in 1:24
  idx = 16(i-1)+1:16*i
  aucs = auc(conc[idx], t[idx], method=:linear)
  aumcs = aumc(conc[idx], t[idx], method=:linear)
  @test_nowarn auc(conc[idx], t[idx], method=:log_linear)
  @test_nowarn aumc(conc[idx], t[idx], method=:log_linear)
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
