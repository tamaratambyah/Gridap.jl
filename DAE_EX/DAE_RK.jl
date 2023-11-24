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
kop = AffineFEOperator(q,l,R,W)
kFE = solve(kop)


function lhs(t,u,v)
  return ∫( u*v )dΩ
end
function rhs(t,u,v,k)
  return ∫(v*f(t))dΩ - ∫(( k*∇(v)⊙∇(u) ))dΩ
end


""" DIAGNOSTIC FE OPERATOR """



# using Gridap.FESpaces
# import Gridap.FESpaces: FEOperatorFromWeakForm
# import Gridap.FESpaces: get_test
# import Gridap.FESpaces: get_trial
# import Gridap.FESpaces: collect_cell_vector
# import Gridap.Algebra: allocate_vector

# """ Allowing for prognostic input """


# # mutable struct DiagnosticOperator <: NonlinearOperator
# # end


# ### FROM FEOPERATORS.jl
# import Gridap.FESpaces: AlgebraicOpFromFEOp
# function Gridap.FESpaces.allocate_residual(op::AlgebraicOpFromFEOp,x::AbstractVector)


#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)

#   F = interpolate(20,R)
#   # F = EvaluationFunction(trial,op.F)

#   allocate_residual(op.feop,u,F)
# end

# function Gridap.FESpaces.residual!(b::AbstractVector,op::AlgebraicOpFromFEOp,x::AbstractVector)
#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)
#   residual!(b,op.feop,u,F)
# end

# function Gridap.FESpaces.residual(op::AlgebraicOpFromFEOp,x::AbstractVector)
#   F = interpolate(20,R)
#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)
#   residual(op.feop,u,F)
# end

# function Gridap.FESpaces.allocate_jacobian(op::AlgebraicOpFromFEOp,x::AbstractVector)
#   F = interpolate(20,R)
#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)
#   allocate_jacobian(op.feop,u,F)
# end

# function Gridap.FESpaces.jacobian!(A::AbstractMatrix,op::AlgebraicOpFromFEOp,x::AbstractVector)
#   F = interpolate(20,R)
#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)
#   jacobian!(A,op.feop,u,F)
# end

# function Gridap.FESpaces.jacobian(op::AlgebraicOpFromFEOp,x::AbstractVector)
#   F = interpolate(20,R)
#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)
#   jacobian(op.feop,u,F)
# end

# function Gridap.FESpaces.residual_and_jacobian!(b::AbstractVector,A::AbstractMatrix,op::AlgebraicOpFromFEOp,x::AbstractVector)
#   F = interpolate(20,R)
#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)
#   residual_and_jacobian!(b,A,op.feop,u,F)
# end

# function Gridap.FESpaces.residual_and_jacobian(op::AlgebraicOpFromFEOp,x::AbstractVector)
#   F = interpolate(20,R)
#   trial = get_trial(op.feop)
#   u = EvaluationFunction(trial,x)
#   residual_and_jacobian(op.feop,u,F)
# end

# #### FEOM FEOPERATORSFROMWEAKFORM.jl
# mutable struct DiagnosticFEOperator <: FEOperator
#   res::Function
#   jac::Function
#   trial::FESpace
#   test::FESpace
#   assem::Assembler
# end

# function DiagnosticFEOperator(
#   res::Function,jac::Function,trial::FESpace,test::FESpace)
#   assem = SparseMatrixAssembler(trial,test)
#   DiagnosticFEOperator(res,jac,trial,test,assem)
# end

# get_test(feop::DiagnosticFEOperator) = feop.test

# get_trial(feop::DiagnosticFEOperator) = feop.trial

# get_matrix(feop::DiagnosticFEOperator) = get_matrix(feop.op)

# get_vector(feop::DiagnosticFEOperator) = get_vector(feop.op)

# get_algebraic_operator(feop::DiagnosticFEOperator) = feop.op

# function allocate_residual(op::DiagnosticFEOperator,uh,F)
#   V = get_test(op)
#   v = get_fe_basis(V)
#   vecdata = collect_cell_vector(V,op.res(uh,v,F))
#   allocate_vector(op.assem, vecdata)
# end

# function residual!(b::AbstractVector,op::DiagnosticFEOperator,uh,F)
#   V = get_test(op)
#   v = get_fe_basis(V)
#   vecdata = collect_cell_vector(V,op.res(uh,v,F))
#   assemble_vector!(b,op.assem, vecdata)
#   b
# end

# function allocate_jacobian(op::DiagnosticFEOperator,uh,F)
#   U = get_trial(op)
#   V = get_test(op)
#   du = get_trial_fe_basis(U)
#   v = get_fe_basis(V)
#   matdata = collect_cell_matrix(U,V,op.jac(uh,du,v,F))
#   allocate_matrix(op.assem, matdata)
# end

# function jacobian!(A::AbstractMatrix,op::DiagnosticFEOperator,uh,F)
#   U = get_trial(op)
#   V = get_test(op)
#   du = get_trial_fe_basis(U)
#   v = get_fe_basis(V)
#   matdata = collect_cell_matrix(U,V,op.jac(uh,du,v,F))
#   assemble_matrix!(A,op.assem,matdata)
#   A
# end

# function residual_and_jacobian!(
#   b::AbstractVector,A::AbstractMatrix,op::DiagnosticFEOperator,uh,F)
#   U = get_trial(op)
#   V = get_test(op)
#   du = get_trial_fe_basis(U)
#   v = get_fe_basis(V)
#   data = collect_cell_matrix_and_vector(U,V,op.jac(uh,du,v,F),op.res(uh,v,F))
#   assemble_matrix_and_vector!(A, b, op.assem, data)
#   (b,A)
# end

# function residual_and_jacobian(op::DiagnosticFEOperator,uh,F)

#   U = get_trial(op)
#   V = get_test(op)
#   du = get_trial_fe_basis(U)
#   v = get_fe_basis(V)
#   data = collect_cell_matrix_and_vector(U,V,op.jac(uh,du,v,F),op.res(uh,v,F))
#   A, b = assemble_matrix_and_vector(op.assem, data)
#   (b, A)
# end



# ### DIAGNOSTIC SOLVE
# res_k(kk,v,F) = ∫(kk*v)dΩ - ∫( k*v*F )dΩ
# jac_k(kk,dkk,v,F) = ∫( dkk*v )dΩ
# kop = DiagnosticFEOperator(res_k,jac_k,R,W)

# kFE = solve(kop)
# println(get_free_dof_values(kFE))

# get_trial(kop)


"""solve step"""

import Gridap.FESpaces: get_algebraic_operator
import Gridap.ODEs.ODETools: allocate_cache

### EX-RK
op = EXRKDAE(lhs,rhs,U,V)

a = reshape([0.0],1,1)
b = [1.0]
c = [0.0]
ls = LUSolver()
odeop = get_algebraic_operator(op)
t0 = 0.0
dt = 0.001


ode_cache = allocate_cache(odeop)
vi = similar(u0)
ki = [similar(u0)]
M = allocate_jacobian(op,0.0,uh0,nothing)
get_mass_matrix!(M,odeop,t0,u0,ode_cache)
l_cache = nothing

lop = DAERKStageOperator(odeop,t0,dt,u0,ode_cache,vi,ki,0,a,M)

uf = similar(u0)



i = 1
  ti = t0 + c[i]*dt
  ode_cache = update_cache!(ode_cache,odeop,ti)
  update!(lop,ti,ki[i],i)
  l_cache = solve!(uf,ls,lop,l_cache)
  update!(lop,ti,uf,i)

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
