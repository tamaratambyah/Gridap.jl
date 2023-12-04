using Pkg

using Gridap
using Gridap.Algebra, Gridap.ODEs.ODETools, Gridap.ODEs.TransientFETools
using LinearAlgebra

# using Plots
# using Test
# using GridapSolvers

# initial condition

include("DAE_transient_operator.jl")
include("transient_jacobians.jl")
include("DAERKStageOperator.jl")
include("DiagnosticFEOperatorFromWeakForm.jl")
include("DAEOperator.jl")
# include("diagnostic_operator.jl")

k = 4.0

u(x,t) = x[1]*(1-x[1])*t
u(t) = x -> u(x,t)
∂tu = ∂t(u)
f(t) = x -> ∂t(u)(x,t)-k*Δ(u(t))(x)


n = 2
p = 2
degree = 4*(p+1)
L = 1
dx = L/n

domain = (0.0, L)
partition = (n )
model  = CartesianDiscreteModel(domain,partition)#, isperiodic=(true,true))
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

V = TestFESpace(model,
                ReferenceFE(lagrangian,Float64,p),
                conformity=:H1,
                dirichlet_tags="boundary")
U = TransientTrialFESpace(V,u)

# space for k
W = TestFESpace(model,
                ReferenceFE(lagrangian,Float64,1),
                conformity=:H1)
R = TransientTrialFESpace(W)


q(uu,v) = ∫(uu*v)dΩ
l(v) = ∫( u(0.0)*v)dΩ
uop = AffineFEOperator(q,l,U(0.0),V)
uh0 = solve(uop)
u0 = get_free_dof_values(uh0)


q(kk,v) = ∫(kk*v)dΩ
l(v) = ∫( k*v )dΩ
res(kk,v) = q(kk,v) - l(v)
jac(kk,dk,v) = q(dk,v)
kop = FEOperator(res,jac,R,W)
kFE = solve(kop)
k0 = get_free_dof_values(kFE)


function lhs(t,u,v)
  return ∫( u*v )dΩ
end
function rhs(t,u,v,k)
  return ∫(v*f(t))dΩ - ∫(( k*∇(v)⊙∇(u) ))dΩ
end



using Gridap.FESpaces
import Gridap.FESpaces: FEOperatorFromWeakForm
import Gridap.FESpaces: get_test
import Gridap.FESpaces: get_trial
import Gridap.FESpaces: collect_cell_vector
import Gridap.Algebra: allocate_vector
import Gridap.FESpaces: AlgebraicOpFromFEOp
import Gridap.FESpaces: get_algebraic_operator
import Gridap.ODEs.ODETools: allocate_cache

using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())


"""solve step"""

res_k(kk,v,F) = q(kk,v) - ∫( k*v )dΩ
jac_k(kk,dk,v,F) = q(dk,v)

### EX-RK
op = EXRKDAE(lhs,rhs,U,V)

a = reshape([0.0],1,1)

b = [1.0]
c = [0.0]
ls = LUSolver()
odeop = get_algebraic_operator(op)
t0 = 0.0
dt = 0.001

F0 = [20.0,20.0,20.0]
kop = DiagnosticFEOperator(res_k,jac_k,R,W)
feop = get_algebraic_operator(kop)



y0 = copy(k0)

ode_cache = allocate_cache(odeop)
vi = similar(u0)
ki = [similar(u0)]
M = allocate_jacobian(op,0.0,uh0,nothing)
get_mass_matrix!(M,odeop,t0,u0,ode_cache)
l_cache = nothing

nl_cache = nothing

lop = DAERKStageOperator(odeop,t0,dt,u0,ode_cache,vi,ki,0,a,M,y0,R)
nlop = AlgebraicDAEOpFromFEOp(feop.feop,F0,U(0.0))



uf = similar(u0)
kf = similar(k0)





i = 1
  ti = t0 + c[i]*dt
  ode_cache = update_cache!(ode_cache,odeop,ti)
  update!(lop,ti,ki[i],i,k0)
  l_cache = solve!(uf,ls,lop,l_cache)
  update!(lop,ti,uf,i,k0)

  # solve for u - required for SW

  # solve for diagnostic variable
  solve!(kf,nls,nlop,nl_cache)

  # update RK operator
  update!(lop,ti,uf,i,kf) # store in RK operator


@. uf = u0
@. uf = uf + dt*b[i]*lop.ki[i]
tf = t0 + dt

# error
u_ex = get_free_dof_values( interpolate_everywhere(u(tf),U(tf)))
println(uf)
println(u_ex)
println(abs(sum(uf-u_ex)))

# prepare next time iteration
@. u0 = u_ex
t0 = tf
@. k0 = kf
