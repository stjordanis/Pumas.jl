using PuMaS, CSV, LabelledArrays, Distributions, PDMats

data = process_nmtran(CSV.read(joinpath(dirname(pathof(PuMaS)),"..","examples/data1.csv")),
                    [:sex,:wt,:etn])

for subject in data.subjects
    obs1 = subject.observations[1]
    if obs1.time == 0
        subject.observations[1] = PuMaS.Observation(sqrt(eps()), obs1.val, obs1.cmt)
    end
end

### Function-Based Interface

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              a = ConstDomain(0.2)))

function rfx_f(p)
    RandomEffectSet((η=MvNormal(p.Ω),))
end

function col_f(p,rfx,cov)
   (Ka = t->t*p.θ[1],  # pre
    CL = p.θ[2] * ((cov.wt/70)^0.75) *
         (p.θ[4]^cov.sex) * exp(rfx.η[1]),
    V  = p.θ[3] * exp(rfx.η[2]))
end

@SLVector OneCompartmentVector Float64 [Depot,Central]

function init_f(col,t0)
    OneCompartmentVector(0.0,0.0)
end

function onecompartment_f(u,p,t)
    OneCompartmentVector(-p.Ka(t)*u[1],
                          p.Ka(t)*u[1] - (p.CL/p.V)*u[2])
end

function post_f(col,u,t)
    conc = u[2] / col.V
    (dv = Normal(conc, conc*col.Σ),)
end

function derived_f(col,sol,obstimes,obs)
    (obs_cmax = maximum(obs[:dv]),)
end

mobj = PKPDModel(p,rfx_f,col_f,init_f,onecompartment_f,post_f,derived_f)

x0 = (θ = [2.268,74.17,468.6,0.5876],
      Ω = PDMat([0.05 0.0;
                 0.0 0.2]),
      σ = 0.1)
subject1 = data.subjects[1]
y0 = init_random(mobj, x0)

sol_mobj   = solve(mobj,subject1,x0,y0)

## DSL

@test_broken begin

  m_diffeq = @model begin
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
          Ka = t -> t*θ[1]
          CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
          V  = θ[3] * exp(η[2])
      end

      @dynamics begin
          cp       =  Central/V
          Depot'   = -Ka(t)*Depot
          Central' =  Ka(t)*Depot - CL*cp
      end

      @post begin
          conc = Central / V
          dv ~ Normal(conc, conc*σ)
      end
  end
end
