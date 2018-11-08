using PuMaS, ForwardDiff, DiffEqDiffTools, Test, Random, LabelledArrays

AD_gradient = ForwardDiff.gradient
AD_hessian = ForwardDiff.hessian
FD_gradient = DiffEqDiffTools.finite_difference_gradient
FD_jacobian = DiffEqDiffTools.finite_difference_jacobian
FD_hessian = function (f, x)
    grad_fun = y -> FD_gradient(f, y)
    FD_jacobian(grad_fun, x, Val{:central}, eltype(x), Val{false})
end

@testset "Derivatives" begin
    data = process_nmtran(example_nmtran_data("data1"),
                        [:sex,:wt,:etn])
    subject = data.subjects[1]
    # Cut off the `t=0` pre-dose observation as it throws conditional_loglikelihood calculations
    # off the scale (variance of the simulated distribution is too small).
    for subject in data.subjects
        obs1 = subject.observations[1]
        if obs1.time == 0
            popfirst!(subject.observations)
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

        @dynamics begin
            cp       =  Central/V
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
    rfx_f(p) = RandomEffectSet((η=MvNormal(p.Ω),))
    function col_f(p,rfx,cov)
        (Ka = p.θ[1],
        CL = p.θ[2] * ((cov.wt/70)^0.75) * (p.θ[4]^cov.sex) * exp(rfx.η[1]),
        V  = p.θ[3] * exp(rfx.η[2]),
        σ = p.σ)
    end
    init_f(col,t0) = @LArray [0.0, 0.0] (:Depot, :Central)
    function onecompartment_f(du,u,p,t)
        cp = u.Central/p.V
        du.Depot = -p.Ka*u.Depot
        du.Central = p.Ka*u.Depot - p.CL*cp
    end
    prob = ODEProblem(onecompartment_f,nothing,nothing,nothing)
    function derived_f(col,sol,obstimes)
        central = map(x->x.Central, sol)
        _conc = @. central / col.V
        _cmax = maximum(_conc)
        _dv = @. Normal(_conc, _conc*col.σ)
        (conc = _conc, cmax = _cmax), (dv = _dv,)
    end
    model_ip = PKPDModel(p,rfx_f,col_f,init_f,prob,derived_f)

    # Initial data
    θ0 = [2.268,74.17,468.6,0.5876]
    x0 = (θ = θ0, Ω = PDMat([0.05 0.0; 0.0 0.2]), σ = 0.1)
    Random.seed!(0); y0 = init_random(model, x0)

    # Test gradient and hessian of an observable w.r.t. θ
    function test_obs(model)
        function (θ)
            _x0 = (θ = θ, Ω = x0.Ω, σ = x0.σ)
            sim = simobs(model, subject, _x0, y0; abstol=1e-14, reltol=1e-14)
            sim[:cmax]
        end
    end

    fun = test_obs(model)
    @test FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-8
    @test FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-3

    fun = test_obs(model_ip)
    @test FD_gradient(fun, θ0) ≈ AD_gradient(fun, θ0) atol=1e-8
    @test FD_hessian(fun, θ0) ≈ AD_hessian(fun, θ0) atol=1e-3

    # Test gradient and hessian of the total log conditional_loglikelihood w.r.t. θ
    function test_conditional_loglikelihood(model)
        function (θ)
            _x0 = (θ = θ, Ω = x0.Ω, σ = x0.σ)
            conditional_loglikelihood(model, subject, _x0, y0; abstol=1e-14, reltol=1e-14)
        end
    end

    fun = test_conditional_loglikelihood(model)
    grad_FD, hes_FD = FD_gradient(fun, θ0), FD_hessian(fun, θ0)
    _, _, grad_AD, hes_AD = PuMaS.conditional_loglikelihood_derivatives(
        model,subject,x0,y0,:θ,abstol=1e-14,reltol=1e-14)
    @test grad_FD ≈ grad_AD atol=2e-6
    @test hes_FD ≈ hes_AD atol=5e-3

    fun = test_conditional_loglikelihood(model_ip)
    grad_FD, hes_FD = FD_gradient(fun, θ0), FD_hessian(fun, θ0)
    _, _, grad_AD, hes_AD = PuMaS.conditional_loglikelihood_derivatives(
        model_ip,subject,x0,y0,:θ,abstol=1e-14, reltol=1e-14)
    @test grad_FD ≈ grad_AD atol=2e-6
    @test hes_FD ≈ hes_AD atol=5e-3

    PuMaS.marginal_loglikelihood(model_ip,subject,x0,y0,PuMaS.Laplace())
end

@testset "Magic arguments" begin
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

    subject = process_nmtran(example_nmtran_data("event_data/data2"),
                         [], [:cp])[1]

    θ0 = [1.5, 1.0, 30.0, 5.0]
    x0 = (θ = θ0,)
    y0 = (η = [0.0,0.0],)

    test_fun = function(θ)
        _x0 = (θ = θ,)
        sim = simobs(mlag, subject, _x0, y0; abstol=1e-14, reltol=1e-14)
        sim[:cmax]
    end

    grad_FD = FD_gradient(test_fun, θ0)
    @test_broken grad_AD = AD_gradient(test_fun, θ0)
    # @show grad_FD[4]
    # @show grad_AD[4]
end
