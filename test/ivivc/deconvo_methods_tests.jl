using PuMaS.IVIVC, Test
# Tests for Wagner Nelson method
vivo_pop = @test_nowarn read_vivo("../examples/ivivc_test_data/vivo_data.csv")
vivo_fast = vivo_pop[1]["fast"]
kel = 0.39
@test_nowarn wagner_nelson(vivo_fast.conc, vivo_fast.time, kel)

# Tests for calc_input_rate function

# read reference vivo data
ref_vivo = @test_nowarn read_vivo("../examples/ivivc_test_data/ref_vivo.csv")

vivo_sub = ref_vivo[1]["solution"]

# model the data with bateman function and get ka, kel and V
@test_nowarn vivo_model(vivo_sub, :bat)

ka, kel, V = vivo_sub.pmin

# read vitro data and model with Emax
vitro_sub = @test_nowarn read_vitro("../examples/ivivc_test_data/vitro_data.csv")[1]["fast"]

@test_nowarn vitro_model(vitro_sub, :e)

# Emax model Opt. params (vitro modeling)
p_n = vitro_sub.pmin

# vitro dissolution rate (derivative of Emax model)
r_diss(t) = p_n[1] * p_n[2] * (p_n[3]^p_n[2]) * (t^(p_n[2]-1)) / ((p_n[3]^p_n[2] + t^p_n[2])^2)

# a simple model which directly relates vitro dissolution to vivo conc profile,
# p[1] is amplitude scaling factor and p[2] is time scaling factor
f(c, p, t) = p[1] * r_diss(t * p[2]) / V - kel * c

p_d = @test_nowarn calc_input_rate(vivo_sub.conc, vivo_sub.time, f, 0.0, [0.4, 0.3], box=true, lb=[0.0, 0.0], ub=[Inf, Inf])

