using Test
using PuMaS, Distributions

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

    @collate begin
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

    @post begin
        cHat = max(x2,0.0) / V1
        neutHat = x8 + circ0

        logc ~ Normal(log(cHat), sigma)
        logn ~ Normal(log(neutHat), sigmaNeut)
    end
end

amt = vcat(repeat(vcat(80_000, fill(0, 13)), 2),
           repeat([80_000, 0], 13),
           fill(0, 35))

cObs = [329.316897447047, 697.320965998204, 740.862512133513, 1205.15076668876,
        1257.28101953198, 1185.71416687525, 1169.7063940622, 823.962074043762,
        513.274768773849, 346.136959891888, 276.361387949035, 213.601617020019,
        173.647272514587, 518.208870321164, 721.525972452254, 877.460778124162,
        1641.84491836671, 1524.28557777049, 1328.15152209464, 1027.50974260975,
        1054.73939362761, 588.179898985728, 523.377441410767, 359.041215680208,
        288.530019317251, 223.095571453367, 237.689682139595, 322.570370746109,
        254.3220036859, 338.414939813066, 318.777532513748, 320.96415891391,
        349.35616434716, 332.675778780383, 346.915652936693, 287.961515350368,
        328.163622421969, 456.842872075472, 691.445627838082, 876.984632948753,
        1327.2083658499, 1487.19789701485, 1757.6823642731, 1572.93640961084,
        1457.56674233743, 1258.17056331459, 855.700340358656, 606.121931578173,
        448.847628780838, 449.37026359581, 419.525850513054, 232.016697964733,
        223.40918064203]

evid = vcat(repeat(vcat(1, fill(0, 13)), 2),
             repeat(1:-1:0, 13),
             fill(0, 35))

iObsPD = vcat(2, 30:4:54, 69:89)

iObsPK = vcat(3:14, 16:28, 30:2:54, 55:69)

neutObs = [4.39843397609086, 4.80780714283483, 5.06606895891759, 4.0730742193975,
           5.10514001479531, 4.23599914241832, 3.9396754686384, 3.29277265985221,
           3.01241839309413, 2.99099966839828, 2.6706937317036, 2.23542967876045,
           1.97661857432204, 2.42883377140825, 2.24406478333476, 2.04166713342004,
           2.55189657641634, 2.44636008460013, 3.26380829001017, 3.2265501130707,
           3.67256011264626, 4.34338234964359, 4.55240425991815, 4.44582475775002,
           5.23971625211994, 4.61043939062381, 5.57263663887337, 5.2697689718298,
           6.03135090450531]

time_ = [0, 0, 0.083, 0.167, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 12,
         12.083, 12.167, 12.25, 12.5, 12.75, 13, 13.5, 14, 15, 16, 18, 20, 24,
         24, 36, 36, 48, 48, 60, 60, 72, 72, 84, 84, 96, 96, 108, 108, 120, 120,
         132, 132, 144, 144, 156, 156, 168, 168, 168.083, 168.167, 168.25, 168.5,
         168.75, 169, 169.5, 170, 171, 172, 174, 176, 180, 186, 192, 216, 240,
         264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576,
         600, 624, 648, 672]

cvals = fill(NaN,length(time_))
nvals = fill(NaN,length(time_))
cvals[iObsPK] = cObs
nvals[iObsPD] = neutObs

subject = Subject(obs = [ PuMaS.Observation(t, (logc = log(c), logn = log(n)), 1) for
                              (t,c,n) in zip(time_,cvals,nvals) if
                              !(isnan(n) && isnan(c))],
                  evs = [ PuMaS.Event(Float64(a),t,Int8(id),1) for
                              (t,a,id) in zip(time_,amt,evid) if id != 0])

x0 = PuMaS.init_param(m_neut)

sol_diffeq = solve(m_neut,subject,x0)

# Should be made into a population test
sol_diffeq = solve(m_neut,subject,x0, parallel_type = PuMaS.Threading)

@test !isnan(PuMaS.likelihood(m_neut, subject, x0, ()))
