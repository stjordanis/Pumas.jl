using PuMaS.NCA, Test, CSV
using PuMaS

root = joinpath(dirname(pathof(PuMaS)), "..")
data = CSV.read("$root/examples/nca_test_data/dapa_IV.csv")

conc = Float64.(data[:CObs])
t = Float64.(data[:TIME])

data = CSV.read("$root/examples/nca_test_data/Final_Parameters_Pivoted.csv")

correct_auc = Float64.(data[:AUCINF_obs])
correct_aumc = Float64.(data[:AUMCINF_obs])

test_auc = rand(24,)
test_aumc = rand(24,)

for method in (:linear, :log_linear)
  @inferred AUC(conc, t, method)
  @inferred AUMC(conc, t, method)
end

for i = 0:23
  test_auc[i+1] = AUC(conc[16*i+1:16*i+16], t[16*i+1:16*i+16], :linear)
  test_aumc[i+1] = AUMC(conc[16*i+1:16*i+16], t[16*i+1:16*i+16], :linear)
  @test_nowarn AUC(conc[16*i+1:16*i+16], t[16*i+1:16*i+16], :log_linear)
  @test_nowarn AUMC(conc[16*i+1:16*i+16], t[16*i+1:16*i+16], :log_linear)
end

@test test_auc[1] ≈ correct_auc[1] atol = 1.0e-6
@test test_auc[2] ≈ correct_auc[2] atol = 1.0e-6
@test test_auc[3] ≈ correct_auc[3] atol = 1.0e-6
@test test_auc[4] ≈ correct_auc[4] atol = 1.0e-6
@test test_auc[5] ≈ correct_auc[5] atol = 1.0e-6
@test_broken test_auc[6] ≈ correct_auc[6] atol = 1.0e-6
@test test_auc[7] ≈ correct_auc[7] atol = 1.0e-6
@test test_auc[8] ≈ correct_auc[8] atol = 1.0e-6
@test_broken test_auc[9] ≈ correct_auc[9] atol = 1.0e-6
@test test_auc[10] ≈ correct_auc[10] atol = 1.0e-6
@test test_auc[11] ≈ correct_auc[11] atol = 1.0e-6
@test test_auc[12] ≈ correct_auc[12] atol = 1.0e-6
@test test_auc[13] ≈ correct_auc[13] atol = 1.0e-6
@test test_auc[14] ≈ correct_auc[14] atol = 1.0e-6
@test test_auc[15] ≈ correct_auc[15] atol = 1.0e-6
@test test_auc[16] ≈ correct_auc[16] atol = 1.0e-6
@test test_auc[17] ≈ correct_auc[17] atol = 1.0e-6
@test test_auc[18] ≈ correct_auc[18] atol = 1.0e-6
@test test_auc[19] ≈ correct_auc[19] atol = 1.0e-6
@test test_auc[20] ≈ correct_auc[20] atol = 1.0e-6
@test test_auc[21] ≈ correct_auc[21] atol = 1.0e-6
@test test_auc[22] ≈ correct_auc[22] atol = 1.0e-6
@test test_auc[23] ≈ correct_auc[23] atol = 1.0e-6
@test test_auc[24] ≈ correct_auc[24] atol = 1.0e-6

@test test_aumc[1] ≈ correct_aumc[1] atol = 1.0e-6
@test test_aumc[2] ≈ correct_aumc[2] atol = 1.0e-6
@test test_aumc[3] ≈ correct_aumc[3] atol = 1.0e-6
@test test_aumc[4] ≈ correct_aumc[4] atol = 1.0e-6
@test test_aumc[5] ≈ correct_aumc[5] atol = 1.0e-6
@test_broken test_aumc[6] ≈ correct_aumc[6] atol = 1.0e-6
@test test_aumc[7] ≈ correct_aumc[7] atol = 1.0e-6
@test test_aumc[8] ≈ correct_aumc[8] atol = 1.0e-6
@test_broken test_aumc[9] ≈ correct_aumc[9] atol = 1.0e-6
@test test_aumc[10] ≈ correct_aumc[10] atol = 1.0e-6
@test test_aumc[11] ≈ correct_aumc[11] atol = 1.0e-6
@test test_aumc[12] ≈ correct_aumc[12] atol = 1.0e-6
@test test_aumc[13] ≈ correct_aumc[13] atol = 1.0e-6
@test test_aumc[14] ≈ correct_aumc[14] atol = 1.0e-6
@test test_aumc[15] ≈ correct_aumc[15] atol = 1.0e-6
@test test_aumc[16] ≈ correct_aumc[16] atol = 1.0e-6
@test test_aumc[17] ≈ correct_aumc[17] atol = 1.0e-6
@test test_aumc[18] ≈ correct_aumc[18] atol = 1.0e-6
@test test_aumc[19] ≈ correct_aumc[19] atol = 1.0e-6
@test test_aumc[20] ≈ correct_aumc[20] atol = 1.0e-6
@test test_aumc[21] ≈ correct_aumc[21] atol = 1.0e-6
@test test_aumc[22] ≈ correct_aumc[22] atol = 1.0e-6
@test test_aumc[23] ≈ correct_aumc[23] atol = 1.0e-6
@test test_aumc[24] ≈ correct_aumc[24] atol = 1.0e-6
