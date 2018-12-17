using PuMaS, Test, CSV, Random, Distributions, LinearAlgebra

# Read the data
data = process_nmtran(example_nmtran_data("data1"),
                      [:sex,:wt,:etn])

# Cut off the `t=0` pre-dose observation as it throws conditional_ll calculations
# off the scale (variance of the simulated distribution is too small).
for subject in data.subjects
  obs1 = subject.observations[1]
  if obs1.time == 0
    popfirst!(subject.observations)
  end
end



# Definition using diffeqs
m = @model begin
  @param begin
    θ ~ Constrained(MvNormal([2.268,74.17,468.6,0.5876],
                             Diagonal([4.0, 30.0, 100.0, 0.5])),
    lower=zeros(4), init=ones(4))
    Ω ~ InverseWishart(5.0, 2.0*[0.05 0.0;
                                 0.0 0.2])
    σ ~ Gamma(1,0.1)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates sex wt etn

  @pre begin
    Ka = θ[1]
    CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
    V  = θ[3] * exp(η[2])
  end

  @dynamics OneCompartmentModel

  @derived begin
    conc = @. Central / V
    dv ~ @. Normal(conc, conc*σ)
  end
end



t_param = PuMaS.totransform(m.param)
t_random = PuMaS.totransform(m.random(init_param(m)))

using TransformVariables
t = as((t_param, t_random))


struct BayesModel{M,D}
  model::M
  data::D
end

function (b::BayesModel)((v_param, v_random))
  s_param = b.model.param
  l_param = sum(map(logpdf, s_param.params, v_param))
  l_param - PuMaS.penalized_conditional_nll(b.model, b.data.subjects[1], v_param, v_random)
end

b = BayesModel(m, data)

v_param = (θ = [2.268,74.17,468.6,0.5876],
           Ω = cholesky([0.05 0.0;
                         0.0 0.2]),
           σ = 0.1)

v_random = init_random(m, v_param)

b((v_param, v_random))


using DynamicHMC, LogDensityProblems
P = TransformedLogDensity(t, b)
∇P = ADgradient(:ForwardDiff, P)
chain,tuned = NUTS_init_tune_mcmc(∇P, 10_000)

[TransformVariables.transform(t, get_position(c))[1].θ[2] for c in chain]

# check sampling from constrained distributions
x = [TransformVariables.transform(t, get_position(c))[1].θ[1] for c in chain]
u = filter(z -> z > 0, rand(Normal(2.268, 2.0), 100_000))
@test mean(x) ≈ mean(u) rtol=0.01
@test var(x) ≈ var(u) rtol=0.01





using PuMaS, Test, CSV, Random, Distributions, LinearAlgebra


theopp = process_nmtran(example_nmtran_data("event_data/THEOPP"),[:WT,:SEX])

theopmodel_bayes = @model begin
    @param begin
        θ ~ Constrained(MvNormal([1.9,0.0781,0.0463,1.5,0.4],
                                 Diagonal([9025, 15.25, 5.36, 5625, 400])),
                        lower=[0.1,0.008,0.0004,0.1,0.0001],
                        upper=[5,0.5,0.09,5,1.5],
                        init=[1.9,0.0781,0.0463,1.5,0.4]
                        )
      
        Ω ~ InverseWishart(2, fill(0.9,1,1))
        σ = 0.388
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Ka = (SEX == 1 ? θ[1] : θ[4]) + η[1]
        K = θ[2]
        CL  = θ[3]*(WT/70)^θ[5]
        V = CL/K/(WT/70)
    end

    @covariates SEX WT

    @vars begin
        conc = Central / V
    end

    @dynamics OneCompartmentModel

    @derived begin
        dv ~ @. Normal(conc,sqrt(σ))
    end
end


bm = PuMaS.BayesModel(theopmodel_bayes, theopp)

using DynamicHMC, LogDensityProblems

adbm = ADgradient(:ForwardDiff, bm)

chain,tuned = NUTS_init_tune_mcmc(adbm, 100)
