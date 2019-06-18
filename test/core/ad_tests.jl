using PuMaS, ForwardDiff, DiffEqDiffTools, Test, Random, LabelledArrays, DiffResults

AD_gradient = ForwardDiff.gradient
AD_hessian = ForwardDiff.hessian
FD_gradient = DiffEqDiffTools.finite_difference_gradient
FD_jacobian = DiffEqDiffTools.finite_difference_jacobian
FD_hessian = function (f, x)
    grad_fun = y -> FD_gradient(f, y)
    FD_jacobian(grad_fun, x, Val{:central}, eltype(x), Val{false})
end

@testset "Derivatives w.r.t regular parameters" begin
    data = read_pumas(example_nmtran_data("data1"),
                        [:sex,:wt,:etn])
    subject = data[1]
    # Cut off the `t=0` pre-dose observation as it throws conditional_nll calculations
    # off the scale (variance of the simulated distribution is too small).
    for subject in data
        if subject.time[1] == 0
            popfirst!(subject.time)
            popfirst!(subject.observations.dv)
        end
    end

    # Simple one-compartment model (uses static vector)
    model = @model begin
        @param begin
            θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
            Ω ∈ PSDDomain(2)
            σ ∈ RealDomain(lower=0.0, init=1.0)
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

        @vars begin
            cp = Central/V
        end

        @dynamics begin
            Depot'   = -Ka*Depot
            Central' =  Ka*Depot - CL*cp
        end

        @derived begin
            conc = @. Central / V
            cmax = maximum(conc)
            dv ~ @. Normal(conc, conc*σ)
        end
    end

    # Function-Based Interface (non-static vector)
    p = ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),
                Ω = PSDDomain(2),
                σ = RealDomain(lower=0.0, init=1.0)))
    rfx_f(p) = ParamSet((η=MvNormal(p.Ω),))
    function col_f(param,randeffs,subject)
        cov = subject.covariates
        (Ka = param.θ[1],
        CL = param.θ[2] * ((cov.wt/70)^0.75) * (param.θ[4]^cov.sex) * exp(randeffs.η[1]),
        V  = param.θ[3] * exp(randeffs.η[2]),
        σ = param.σ)
    end
    init_f(col,t0) = @LArray [0.0, 0.0] (:Depot, :Central)
    function onecompartment_f(du,u,p,t)
        cp = u.Central/p.V
        du.Depot = -p.Ka*u.Depot
        du.Central = p.Ka*u.Depot - p.CL*cp
    end
    prob = ODEProblem(onecompartment_f,nothing,nothing,nothing)
    function derived_f(col,sol,obstimes,subject)
        central = sol(obstimes;idxs=2)
        conc = @. central / col.V
        cmax = maximum(conc)
        (conc = conc, cmax = cmax, dv = @. Normal(conc, conc*col.σ))
    end
    model_ip = PuMaSModel(p,rfx_f,col_f,init_f,prob,derived_f)

    # Initial data
    θ₀ = [2.268,74.17,468.6,0.5876]
    param = (θ = θ₀,
             Ω = [0.05 0.0;
                  0.0  0.2],
             σ = 0.1)
    Random.seed!(0)
    randeffs = init_randeffs(model, param)

    # Test gradient and hessian of an observable w.r.t. θ
    function test_obs(model)
        function (θ)
            _param = (θ = θ, Ω = param.Ω, σ = param.σ)
            sim = simobs(model, subject, _param, randeffs; abstol=1e-14, reltol=1e-14)
            sim[:cmax]
        end
    end

    fun = test_obs(model)
    @test FD_gradient(fun, θ₀) ≈ AD_gradient(fun, θ₀) atol=1e-8
    @test FD_hessian(fun, θ₀) ≈ AD_hessian(fun, θ₀) atol=1e-3

    fun = test_obs(model_ip)
    @test FD_gradient(fun, θ₀) ≈ AD_gradient(fun, θ₀) atol=1e-8
    @test FD_hessian(fun, θ₀) ≈ AD_hessian(fun, θ₀) atol=1e-3

    # Test gradient and hessian of the total conditional_nll w.r.t. θ
    function test_conditional_nll(model)
        function (θ)
            _param = (θ = θ, Ω = param.Ω, σ = param.σ)
            conditional_nll(model, subject, _param, randeffs; abstol=1e-14, reltol=1e-14)
        end
    end

    fun = test_conditional_nll(model)
    grad_FD, hes_FD = FD_gradient(fun, θ₀), FD_hessian(fun, θ₀)
    grad_AD, hes_AD = ForwardDiff.gradient(fun, θ₀), ForwardDiff.hessian(fun, θ₀)
    @test grad_FD ≈ grad_AD atol=2e-6
    @test hes_FD  ≈ hes_AD  atol=5e-3

    fun = test_conditional_nll(model_ip)
    grad_FD, hes_FD = FD_gradient(fun, θ₀), FD_hessian(fun, θ₀)
    grad_AD, hes_AD = ForwardDiff.gradient(fun, θ₀), ForwardDiff.hessian(fun, θ₀)
    @test grad_FD ≈ grad_AD atol=2e-6
    @test hes_FD  ≈ hes_AD  atol=5e-3

    PuMaS.marginal_nll(model_ip,subject,param,PuMaS.LaplaceI())
end

@testset "Magic argument - lags" begin
    # Test 3 of template_model_ev_system.jl
    mlag = @model begin
        @param    θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        @random   η ~ MvNormal(Matrix{Float64}(I, 2, 2))

        @pre begin
            Ka = θ[1]
            CL = θ[2]*exp(η[1])
            V  = θ[3]*exp(η[2])
            lags = θ[4]
        end

        @dynamics begin
            Depot'   = -Ka*Depot
            Central' =  Ka*Depot - (CL/V)*Central
        end

        @derived begin
            cp = @. Central / V
            cmax = maximum(cp)
        end
    end

    subject = read_pumas(example_nmtran_data("event_data/data2"),
                         [], [:cp])[1]

    θ₀ = [1.5, 1.0, 30.0, 5.0]
    param = (θ = θ₀,)
    randeffs = (η = [0.0,0.0],)

    test_fun = function(θ)
        _param = (θ = θ,)
        sim = simobs(mlag, subject, _param, randeffs; abstol=1e-14, reltol=1e-14)
        sim[:cmax]
    end

    grad_FD = FD_gradient(test_fun, θ₀)
    grad_AD = AD_gradient(test_fun, θ₀)
    @test_broken grad_FD[4] ≈ grad_AD[4] # is NaN
end

@testset "Magic argument - bioav" begin
    # Test 5 of template_model_ev_system.jl
    mbioav = @model begin
        @param    θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        @random   η ~ MvNormal(Matrix{Float64}(I, 2, 2))

        @pre begin
            Ka = θ[1]
            CL = θ[2]*exp(η[1])
            V  = θ[3]*exp(η[2])
            bioav = θ[4]
        end

        @dynamics begin
            Depot'   = -Ka*Depot
            Central' =  Ka*Depot - (CL/V)*Central
        end

        @derived begin
            cp = @. Central / V
            cmax = maximum(cp)
        end
    end

    subject = read_pumas(example_nmtran_data("event_data/data5"),
                         [], [:cp])[1]

    θ₀ = [1.5, 1.0, 30.0, 0.412]
    param = (θ = θ₀,)
    randeffs = (η = [0.0,0.0],)

    test_fun = function(θ)
        _param = (θ = θ,)
        sim = simobs(mbioav, subject, _param, randeffs; abstol=1e-14, reltol=1e-14)
        sim[:cmax]
    end

    grad_FD = FD_gradient(test_fun, θ₀)
    grad_AD = AD_gradient(test_fun, θ₀)
    @test_broken grad_FD[4] ≈ grad_AD[4] # is NaN
end
