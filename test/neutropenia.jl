using Test
using PuMaS, DataFrames

# Example from
# https://github.com/stan-dev/stancon_talks/tree/master/2017/Contributed-Talks/05_margossian/models/neutropenia

m_neut = @model begin

    @param begin

        CL ∈ RealDomain(lower=5.0,  upper=15.0, init=10.0)
        Q  ∈ RealDomain(lower=5.0,  upper=40.0, init=15.0)
        V1 ∈ RealDomain(lower=15.0, upper=70.0, init=35.0)
        V2 ∈ RealDomain(lower=30.0, upper=250.0, init=105.0)

        sigma ∈ RealDomain(lower=1e-8, upper=30.0, init=1.0)

        mtt ∈ RealDomain(lower=30.0,     upper=250.0, init=125.0)
        circ0 ∈ RealDomain(lower=2.0,    upper=12.0,  init=5.0)
        alpha ∈ RealDomain(lower=0.0002, upper=0.02,  init=0.002)
        gamma ∈ RealDomain(lower=0.08,    upper=0.4,   init=0.17)

        sigmaNeut ∈ RealDomain(lower=1e-8, upper=30.0, init=1.0)

        ka ∈ RealDomain(lower=1e-8,  upper=100.0,init=1.0) # no prior defined?
    end

    @pre begin
        k10 = CL / V1
        k12 = Q / V1
        k21 = Q / V2

        ktr = 4 / mtt

        ϵ = eps()
    end

    @dynamics begin
        x1' = -ka * x1
        x2' = ka * x1 - (k10 + k12) * x2 + k21 * x3
        x3' = k12 * x2 - k21 * x3

        # conc = x1 / V1
        # EDrug = alpha * conc
        # prol = x4 + circ0
        # transit1 = x5 + circ0
        # transit2 = x6 + circ0
        # transit3 = x7 + circ0
        # circ =     max(x8 + circ0, eps())  # Device for implementing a modeled initial condition

        # dx4 = ktr * (x4 + circ0) * ((1 - alpha * (x1 / V1) ) * ((circ0 / max(x8 + circ0, ϵ))^gamma) - 1)
        x4' = ktr * (x4 + circ0) * ((1 - alpha * (x1 / V1) ) * ((circ0 / (x8 + circ0))^gamma) - 1)
        x5' = ktr * (x4 - x5)
        x6' = ktr * (x5 - x6)
        x7' = ktr * (x6 - x7)
        # dx8 = ktr * ((x7 + circ0) - max(x8 + circ0, ϵ))
        x8' = ktr * (x7 - x8)
    end

    @derived begin
        cHat = @. max(x2,0.0) / V1
        neutHat = @. x8 + circ0

        logc ~ @. Normal(log(cHat), sigma)
        logn ~ @. Normal(log(neutHat), sigmaNeut)
    end
end

data = DataFrame(time = append!([0.083, 0.167, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
                                 3.0, 4.0, 6.0, 8.0, 12.0, 12.083, 12.167,
                                 12.25, 12.5, 12.75, 13.0, 13.5, 14.0, 15.0,
                                 16.0, 18.0, 20.0, 24.0, 36.0, 48.0, 60.0, 72.0,
                                 84.0, 96.0, 108.0, 120.0, 132.0, 144.0, 156.0,
                                 168.0, 168.083, 168.167, 168.25, 168.5, 168.75,
                                 169.0, 169.5, 170.0, 171.0, 172.0, 174.0,
                                 176.0, 180.0, 186.0, 192.0],
                                 0:24:672),
                  variable = vcat(fill(:logc, 53), fill(:logn, 29)),
                  value = vcat([5.79702, 6.54725, 6.60782, 7.09436, 7.13671,
                                7.0781, 7.06451, 6.71412, 6.24081, 5.84683,
                                5.62171, 5.36411, 5.15703, 6.25038, 6.58137,
                                6.77703, 7.40358, 7.32928, 7.19154, 6.93489,
                                6.96105, 6.37703, 6.2603, 5.88344, 5.6648,
                                5.4076, 5.47097, 5.77632, 5.5386, 5.82427,
                                5.76449, 5.77133, 5.85609, 5.80717, 5.84908,
                                5.66283, 5.79351, 6.12434, 6.53878, 6.77649,
                                7.19083, 7.30465, 7.47175, 7.3607, 7.28452,
                                7.13741, 6.75192, 6.40708, 6.10668, 6.10785,
                                6.03913, 5.44681, 5.409],
                                [1.48125, 1.57024, 1.62257, 1.4044, 1.63025,
                                1.44362, 1.3711, 1.19173, 1.10274, 1.09561,
                                0.982338, 0.804433, 0.681388, 0.887411,
                                0.808289, 0.713767, 0.936837, 0.894601, 1.18289,
                                1.17141, 1.30089, 1.46865, 1.51566, 1.49197,
                                1.65627, 1.52832, 1.71787, 1.66199, 1.79697])) |>
    (data -> unstack(data, :variable, :value))
foreach(col -> replace!(last(col), missing => NaN), eachcol(data))

subject = Subject(obs = data,
                  evs = DosageRegimen(8e4, addl = 14, ii = 12))

x0 = PuMaS.init_param(m_neut)

sol_diffeq = solve(m_neut,subject,x0)

# Should be made into a population test
sol_diffeq = solve(m_neut,subject,x0, parallel_type = PuMaS.Threading)

@test PuMaS.conditional_loglikelihood(m_neut, subject, x0, ()) ≈ -79.54079056760992
