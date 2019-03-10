using PuMaS, Measurements, LabelledArrays
using Random, LinearAlgebra, Test

data = process_nmtran(example_nmtran_data("data1"), [:sex,:wt,:etn])
subject = data.subjects[1]

@testset "Static Vector" begin
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
        end
    end

    # Initial data
    θ0 = [2.268, 74.17, 468.6, 0.5876]
    x0 = (θ = θ0, Ω = PDMat([0.05 0.0; 0.0 0.2]), σ = 0.1)
    Random.seed!(0); y0 = init_random(model, x0)

    # Introduce measurement uncertainty to θ[1] (Ka)
    θ_ms = θ0 .± [0.2, 0.0, 0.0, 0.0]
    x0_ms = (θ=θ_ms, Ω=x0.Ω, σ=x0.σ)
    sol = solve(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    @test sol.u[1] isa SLArray && eltype(sol.u[1]) == Measurement{Float64} # test type stability

    # Compare with manual tracking of uncertainties
    θ_plus = θ0 .+ [0.2, 0.0, 0.0, 0.0]
    θ_minus = θ0 .- [0.2, 0.0, 0.0, 0.0]
    x0_plus = (θ=θ_plus, Ω=x0.Ω, σ=x0.σ)
    x0_minus = (θ=θ_minus, Ω=x0.Ω, σ=x0.σ)
    sim = simobs(model, subject, x0, y0; abstol=1e-14, reltol=1e-14)
    sim_plus = simobs(model, subject, x0_plus, y0; abstol=1e-14, reltol=1e-14)
    sim_minus = simobs(model, subject, x0_minus, y0; abstol=1e-14, reltol=1e-14)
    ## This is only a crude estimate
    err_manual = max(abs(sim[:cmax] - sim_plus[:cmax]), abs(sim[:cmax] - sim_minus[:cmax]))

    sim_ms = simobs(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    err = sim_ms[:cmax].err
    @test err ≈ err_manual atol=1e-2
end

@testset "Non-static vector" begin
    # Function-Based Interface (non-static vector)
    p = ParamSet((θ = VectorDomain(3, lower=zeros(3), init=ones(3)),
                Ω = PSDDomain(2),
                σ = RealDomain(lower=0.0, init=1.0)))
    rfx_f(p) = ParamSet((η=MvNormal(p.Ω),))
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
        (conc = _conc, cmax = _cmax)
    end
    observed_f(col,sol,obstimes,samples) = samples
    model = PKPDModel(p,rfx_f,col_f,init_f,prob,derived_f,observed_f)

    # Initial data
    θ0 = [2.268, 74.17, 468.6, 0.5876]
    x0 = (θ = θ0, Ω = PDMat([0.05 0.0; 0.0 0.2]), σ = 0.1)
    Random.seed!(0); y0 = init_random(model, x0)

    # Introduce measurement uncertainty to θ[1] (Ka)
    θ_ms = θ0 .± [0.2, 0.0, 0.0, 0.0]
    x0_ms = (θ=θ_ms, Ω=x0.Ω, σ=x0.σ)
    sol = solve(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    @test sol.u[1] isa LArray && eltype(sol.u[1]) == Measurement{Float64} # test type stability

    # Compare with manual tracking of uncertainties
    θ_plus = θ0 .+ [0.2, 0.0, 0.0, 0.0]
    θ_minus = θ0 .- [0.2, 0.0, 0.0, 0.0]
    x0_plus = (θ=θ_plus, Ω=x0.Ω, σ=x0.σ)
    x0_minus = (θ=θ_minus, Ω=x0.Ω, σ=x0.σ)
    sim = simobs(model, subject, x0, y0; abstol=1e-14, reltol=1e-14)
    sim_plus = simobs(model, subject, x0_plus, y0; abstol=1e-14, reltol=1e-14)
    sim_minus = simobs(model, subject, x0_minus, y0; abstol=1e-14, reltol=1e-14)
    ## This is only a crude estimate
    err_manual = max(abs(sim[:cmax] - sim_plus[:cmax]), abs(sim[:cmax] - sim_minus[:cmax]))

    sim_ms = simobs(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    err = sim_ms[:cmax].err
    @test err ≈ err_manual atol=1e-2
end

@testset "Magic argument" begin
    # Model with magic argument bioav
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
            bioav = θ[5]
        end

        @dynamics begin
            cp       =  Central/V
            Depot'   = -Ka*Depot
            Central' =  Ka*Depot - CL*cp
        end

        @derived begin
            conc = @. Central / V
            cmax = maximum(conc)
        end
    end

    # Initial data
    θ0 = [2.268, 74.17, 468.6, 0.5876, 0.412]
    x0 = (θ = θ0, Ω = PDMat([0.05 0.0; 0.0 0.2]), σ = 0.1)
    Random.seed!(0); y0 = init_random(model, x0)

    # Introduce measurement uncertainty to θ[1] (Ka)
    θ_ms = θ0 .± [0.0, 0.0, 0.0, 0.0, 0.02]
    x0_ms = (θ=θ_ms, Ω=x0.Ω, σ=x0.σ)
    sol = solve(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    @test sol.u[1] isa SLArray && eltype(sol.u[1]) == Measurement{Float64} # test type stability

    # Compare with manual tracking of uncertainties
    θ_plus = θ0 .+ [0.0, 0.0, 0.0, 0.0, 0.02]
    θ_minus = θ0 .- [0.0, 0.0, 0.0, 0.0, 0.02]
    x0_plus = (θ=θ_plus, Ω=x0.Ω, σ=x0.σ)
    x0_minus = (θ=θ_minus, Ω=x0.Ω, σ=x0.σ)
    sim = simobs(model, subject, x0, y0; abstol=1e-14, reltol=1e-14)
    sim_plus = simobs(model, subject, x0_plus, y0; abstol=1e-14, reltol=1e-14)
    sim_minus = simobs(model, subject, x0_minus, y0; abstol=1e-14, reltol=1e-14)
    ## This is only a crude estimate
    err_manual = max(abs(sim[:cmax] - sim_plus[:cmax]), abs(sim[:cmax] - sim_minus[:cmax]))

    sim_ms = simobs(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    err = sim_ms[:cmax].err
    @test err ≈ err_manual atol=1e-2
end

@testset "Analytic solution" begin
    # Analytic model
    model = @model begin
        @param   θ ∈ VectorDomain(3, lower=zeros(3), init=ones(3))
        @random  η ~ MvNormal(Matrix{Float64}(I, 2, 2))

        @pre begin
            Ka = θ[1]
            CL = θ[2]*exp(η[1])
            V  = θ[3]*exp(η[2])
        end

        @dynamics OneCompartmentModel

        @derived begin
            cp = @. Central / V
            cmax = maximum(cp)
        end
    end

    # Initial data
    θ0 = [1.5, 1.0, 30.0]
    x0 = (θ = θ0,)
    y0 = (η = [0.0,0.0],)

    # Introduce measurement uncertainty to θ[1] (Ka)
    θ_ms = θ0 .± [0.2, 0.0, 0.0]
    x0_ms = (θ=θ_ms,)
    sol = solve(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    @test sol.u[1] isa SLArray && eltype(sol.u[1]) == Measurement{Float64} # test type stability

    # Compare with manual tracking of uncertainties
    θ_plus = θ0 .+ [0.2, 0.0, 0.0]
    θ_minus = θ0 .- [0.2, 0.0, 0.0]
    x0_plus = (θ=θ_plus,)
    x0_minus = (θ=θ_minus,)
    sim = simobs(model, subject, x0, y0; abstol=1e-14, reltol=1e-14)
    sim_plus = simobs(model, subject, x0_plus, y0; abstol=1e-14, reltol=1e-14)
    sim_minus = simobs(model, subject, x0_minus, y0; abstol=1e-14, reltol=1e-14)
    ## This is only a crude estimate
    err_manual = max(abs(sim[:cmax] - sim_plus[:cmax]), abs(sim[:cmax] - sim_minus[:cmax]))

    sim_ms = simobs(model, subject, x0_ms, y0; abstol=1e-14, reltol=1e-14)
    err = sim_ms[:cmax].err
    @test err ≈ err_manual rtol=5e-2
end
