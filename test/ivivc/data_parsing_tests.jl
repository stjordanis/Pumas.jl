using Pumas, Pumas.IVIVC, Test

# read vitro data
vitro_data = read_vitro("../examples/ivivc_test_data/vitro_data.csv")
vitro_subs = vitro_data.subjects

@test typeof(vitro_subs) <: AbstractVector
@test eltype(vitro_subs) <: AbstractDict{String, Pumas.IVIVC.VitroForm}

@test length(vitro_data) == length(vitro_subs)
@test size(vitro_data) == size(vitro_subs)

k = keys(vitro_data[1])
@test "slow" in k
@test "medium" in k
@test "fast" in k

@test vitro_data[1]["slow"].time[1:4] == [0.5, 1.0, 1.5, 2.0]
@test vitro_data[1]["medium"].conc[1:4] == [0.186, 0.299, 0.384, 0.464]
@test vitro_data[1]["fast"].form == "fast"
@test vitro_data[1]["fast"].id == 1
@test vitro_data[1]["slow"].id == 1
@test vitro_data[1]["medium"].id == 1

# read vivo data
vivo_data = read_vivo("../examples/ivivc_test_data/vivo_data.csv")
vivo_subs = vivo_data.subjects

@test typeof(vivo_subs) <: AbstractVector
@test eltype(vivo_subs) <: AbstractDict{String, Pumas.IVIVC.VivoForm}

@test length(vivo_data) == length(vivo_subs)
@test size(vivo_data) == size(vivo_subs)

k = keys(vivo_data[1])
@test "slow" in k
@test "medium" in k
@test "fast" in k

@test vivo_data[1]["slow"].time[1:4] == [0.0, 0.5, 1.0, 1.5]
@test vivo_data[1]["medium"].conc[1:4] == [0.0, 5.182, 26.3, 45.17]
@test vivo_data[1]["fast"].form == "fast"
@test vivo_data[1]["fast"].id == 1
@test vivo_data[1]["slow"].id == 1
@test vivo_data[1]["medium"].id == 1
