using PuMaS, CSV,Test

# avoid rand issue for now
using StatsFuns, ForwardDiff

StatsFuns.RFunctions.poisrand(x::ForwardDiff.Dual) = StatsFuns.RFunctions.poisrand(min(ForwardDiff.value(x),1e6))

df = process_nmtran(example_nmtran_data("sim_poisson"),[:dose])

struct XPoisson{T} <: DiscreteUnivariateDistribution
  λ::T
end

Distributions.rand(d::XPoisson) = 0
function Distributions.logpdf(d::XPoisson, DV::Real)
  LAM = d.λ
  LFAC=DV < 1 ? 0.0 : DV*log(DV)-DV+log(DV*(1+4*DV*(1+2*DV)))/6+log(3.1415)/2
  (-LAM+DV*log(LAM)-LFAC)
end


θ1 = 9.05E-02
θ2 = 4.75E-02
Ω  = 4.74E-02




poisson_model = @model begin
    @param begin
        θ ∈ VectorDomain(2,init=[3.0,0.5],lower=[0.1,0.1])
        Ω ∈ PSDDomain(fill(0.1,1,1))
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        baseline = θ[1]*exp(η[1])
        d50 = θ[2]
    end

    @covariates dose

    @derived begin
        dv ~ @. Poisson(baseline*(1-dose/(dose+d50)))
    end
end


poisson_xmodel = @model begin
    @param begin
        θ ∈ VectorDomain(2,init=[1.02930E+00,4.51854E-01])
        Ω ∈ PSDDomain(fill(4.74E-02,1,1))
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        baseline= θ[1]*exp(η[1])
        d50 = θ[2]
        λ = baseline*(1-dose/(dose+d50))
    end

    @covariates dose

    @derived begin
        dv ~ @. XPoisson(λ)
    end
end

x0 = init_param(poisson_model)
y0 = init_random(poisson_model, x0)

@test solve(poisson_model,df,x0,y0) === nothing
@test simobs(poisson_model,df,x0,y0) != nothing

res = simobs(poisson_model,df,x0,y0)

@test_broken PuMaS.marginal_nll_nonmem(poisson_model,df,x0,y0,Laplace()) ≈ 3809.805993678949562 rtol=1e-6

initial_estimates = [-8.31130E-01,
                     -9.51865E-01,
                     -1.11581E+00,
                     -7.64425E-01,
                     -7.64425E-01,
                     -6.71273E-01,
                     -8.77897E-01,
                     -1.11581E+00,
                     -9.14281E-01,
                     -1.71009E+00,
                     -1.11581E+00,
                     -6.13244E-01,
                     -1.28933E+00,
                     -9.77623E-01,
                     -7.53683E-01,
                     -1.01738E+00,
                     -1.13057E+00,
                     -6.13244E-01,
                     -1.01738E+00,
                     -4.63717E-01]

@testset "initial" begin
  for (i,est) in enumerate(initial_estimates)
    @test PuMaS.rfx_estimate(poisson_model, df[i], x0, Laplace()) ≈ [est] atol=1e-5
  end
  @test 2*PuMaS.marginal_nll(poisson_model,df,x0,Laplace()) ≈ 4015.70427796336 atol=1e-1
end


x1 = (θ=[9.05E-02,  4.75E-02], Ω=fill(4.74E-02,1,1))
