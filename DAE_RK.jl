using Pkg

using Gridap
using Gridap.Algebra, Gridap.ODEs.ODETools, Gridap.ODEs.TransientFETools
using LinearAlgebra

# using Plots
# using Test
# using GridapSolvers

# initial condition
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

""" DAE Transient Operator """

using Gridap.FESpaces
import Gridap.FESpaces: collect_cell_vector
import Gridap.FESpaces: assemble_vector
import Gridap.FESpaces: get_fe_basis

import Gridap.ODEs.TransientFETools: TransientFEOperator
import Gridap.ODEs.TransientFETools: OperatorType
import Gridap.ODEs.TransientFETools: get_test
import Gridap.ODEs.TransientFETools: TransientCellField

struct DAE{C} <: TransientFEOperator{C}
  res::Function
  lhs::Function
  rhs::Function
  jacs::Tuple{Vararg{Function}}
  assem_t::Assembler
  trials::Tuple{Vararg{Any}}
  test::FESpace
  order::Integer
end


function EXRKDAE(lhs::Function,rhs::Function,trial,test)
  res(t,u,v) = lhs(t,u,v) - rhs(t,u,v,k)
  jac(t,u,du,v) = ∫(( du*v ))dΩ
  jac_t(t,u,dut,v) = ∫( dut*v )dΩ
  assem_t = SparseMatrixAssembler(trial,test)
  DAE{Nonlinear}(res,lhs,rhs,(jac,jac_t),assem_t,(trial,∂t(trial)),test,1)
end

Gridap.ODEs.TransientFETools.get_assembler(op::DAE) = op.assem_t

Gridap.ODEs.TransientFETools.get_test(op::DAE) = op.test

Gridap.ODEs.TransientFETools.get_trial(op::DAE) = op.trials[1]

import Gridap.ODEs.TransientFETools: get_order
Gridap.ODEs.TransientFETools.get_order(op::DAE) = op.order

function DAE_rhs!(
  rhs::AbstractVector,
  op::DAE,
  t::Real,
  xh::T,
  yh, # this input needs to be FEFunction from projection
  cache) where T

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.rhs(t,xh,v,yh))
  assemble_vector!(rhs,op.assem_t,vecdata)
  rhs
end

import Gridap.ODEs.TransientFETools: ODEOpFromFEOp
function DAE_rhs!(
  rhs::AbstractVector,
  op::ODEOpFromFEOp,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}}, # this input is tuple of dofs
  yhF::Tuple{Vararg{AbstractVector}}, # this input is tuple of dofs
  ode_cache)
  Xh, = ode_cache   # trial space
  dxh = ()
  dyh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
    dyh = (dyh...,EvaluationFunction(R,yhF[i]))  # force trial space for yh
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  yh=TransientCellField(EvaluationFunction(R,yhF[1]),dyh) # force trial space for yh

  DAE_rhs!(rhs,op.feop,t,xh,yh,ode_cache)
end


""" NOT REQUIRED
import Gridap.ODEs.TransientFETools: rhs!
function Gridap.ODEs.TransientFETools.rhs!(
  rhs::AbstractVector,
  op::DAE,
  t::Real,
  xh::T,
  yh, # this input needs to be FEFunction from projection
  cache) where T

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.rhs(t,xh,v,yh))
  assemble_vector!(rhs,op.assem_t,vecdata)
  rhs
end

function Gridap.ODEs.TransientFETools.rhs!(
  rhs::AbstractVector,
  op::ODEOpFromFEOp,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}},
  yh, # this input needs to be FEFunction from projection
  ode_cache)
  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  rhs!(rhs,op.feop,t,xh,yh,ode_cache)
end
"""


# ## CHECK - use FE functions, rhs! and DAE_rhs! give same output
# _rhs = similar(u0)
# uFE = FEFunction(U(t0),u0)

# DAE_rhs!(_rhs,op,t0,uFE,kFE,nothing)
# println(_rhs)

# rhs!(_rhs,op,t0,uFE,kFE,nothing)
# println(_rhs)

# ## CHECK - use DOFs, rhs! and DAE_rhs! give same output

# kk = get_free_dof_values(kFE)
# xh = (u0,lop.vi)
# yh = (kk,lop.vi)

# DAE_rhs!(_rhs,odeop,t0,xh,yh,ode_cache)
# println(_rhs)

# rhs!(_rhs,lop.odeop,t0,xh,kFE,ode_cache)
# println(_rhs)


# Xh, = ode_cache
# Xh[1]
# xhF = (lop.u0,lop.vi)
# dxh = ()
# dxh = (dxh...,EvaluationFunction(Xh[2],xhF[2]))
# print(get_free_dof_values(dxh[1]))
# xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
# println( get_free_dof_values(xh.cellfield))
# println(u0)

# kk = get_free_dof_values(kFE)
# yhF = (kk,kk)
# dyh = ()
# dyh = (dyh...,EvaluationFunction(R,yhF[2]))
# yh=TransientCellField(EvaluationFunction(R,yhF[1]),dyh)
# println( get_free_dof_values(yh.cellfield))
# println( get_free_dof_values( kFE))

# __rhs = similar(u0)
# DAE_rhs!(__rhs,op,t0,xh,yh,ode_cache)
# println(__rhs)


import Gridap.ODEs.TransientFETools: allocate_residual
function Gridap.ODEs.TransientFETools.allocate_residual(
  op::DAE,
  t0::Real,
  uh::T,
  cache) where T
  V = get_test(op)
  v = get_fe_basis(V)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  vecdata = collect_cell_vector(V,op.res(t0,xh,v))
  allocate_vector(op.assem_t,vecdata)
end

import Gridap.ODEs.TransientFETools: lhs!
function Gridap.ODEs.TransientFETools.lhs!(
  b::AbstractVector,
  op::DAE,
  t::Real,
  xh::T,
  cache) where T
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.lhs(t,xh,v))
  assemble_vector!(b,op.assem_t,vecdata)
  b
end


""" Tranisent Jacobians - allowing for op::DAE """


#### jacobians
import Gridap.ODEs.TransientFETools: fill_initial_jacobians
import Gridap.ODEs.TransientFETools: _matdata_jacobian
import Gridap.ODEs.TransientFETools: fill_jacobians
import Gridap.ODEs.TransientFETools: _vcat_matdata
import Gridap.ODEs.TransientFETools: _matdata_jacobian
import Gridap.ODEs.TransientFETools: jacobians!

function Gridap.ODEs.TransientFETools.allocate_jacobian(
  op::DAE,
  t0::Real,
  uh::CellField,
  cache)
  _matdata_jacobians = fill_initial_jacobians(op,t0,uh)
  matdata = _vcat_matdata(_matdata_jacobians)
  allocate_matrix(op.assem_t,matdata)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::DAE,
  t::Real,
  xh::T,
  i::Integer,
  γᵢ::Real,
  cache) where T
  matdata = _matdata_jacobian(op,t,xh,i,γᵢ)
  assemble_matrix_add!(A,op.assem_t, matdata)
  A
end

function Gridap.ODEs.TransientFETools.jacobians!(
  A::AbstractMatrix,
  op::DAE,
  t::Real,
  xh::TransientCellField,
  γ::Tuple{Vararg{Real}},
  cache)
  _matdata_jacobians = fill_jacobians(op,t,xh,γ)
  matdata = _vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem_t, matdata)
  A
end

function Gridap.ODEs.TransientFETools.fill_initial_jacobians(op::DAE,t0::Real,uh)
  dxh = ()
  for i in 1:get_order(op)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  _matdata = ()
  for i in 1:get_order(op)+1
    _matdata = (_matdata...,_matdata_jacobian(op,t0,xh,i,0.0))
  end
  return _matdata
end

function Gridap.ODEs.TransientFETools.fill_jacobians(
  op::DAE,
  t::Real,
  xh::T,
  γ::Tuple{Vararg{Real}}) where T
  _matdata = ()
  for i in 1:get_order(op)+1
    if (γ[i] > 0.0)
      _matdata = (_matdata...,_matdata_jacobian(op,t,xh,i,γ[i]))
    end
  end
  return _matdata
end

function Gridap.ODEs.TransientFETools._vcat_matdata(_matdata)
  term_to_cellmat_j = ()
  term_to_cellidsrows_j = ()
  term_to_cellidscols_j = ()
  for j in 1:length(_matdata)
    term_to_cellmat_j = (term_to_cellmat_j...,_matdata[j][1])
    term_to_cellidsrows_j = (term_to_cellidsrows_j...,_matdata[j][2])
    term_to_cellidscols_j = (term_to_cellidscols_j...,_matdata[j][3])
  end

  term_to_cellmat = vcat(term_to_cellmat_j...)
  term_to_cellidsrows = vcat(term_to_cellidsrows_j...)
  term_to_cellidscols = vcat(term_to_cellidscols_j...)

  matdata = (term_to_cellmat,term_to_cellidsrows, term_to_cellidscols)
end

function Gridap.ODEs.TransientFETools._matdata_jacobian(
  op::DAE,
  t::Real,
  xh::T,
  i::Integer,
  γᵢ::Real) where T
  Uh = evaluate(get_trial(op),nothing)
  V = get_test(op)
  du = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(Uh,V,γᵢ*op.jacs[i](t,xh,du,v))
end

""" DAE RK OPERATOR """

#####

## DAE RK OPERATOR

######
import Gridap.ODEs.ODETools: RungeKuttaNonlinearOperator
import Gridap.ODEs.ODETools: ODEOperator
mutable struct DAERKStageOperator <: RungeKuttaNonlinearOperator
  odeop::ODEOperator
  ti::Float64
  dt::Float64
  u0::AbstractVector
  ode_cache
  vi::AbstractVector
  ki::AbstractVector
  i::Int
  a::Matrix
  M::AbstractMatrix
end

function Gridap.Algebra.residual!(b::AbstractVector,
  op::DAERKStageOperator,
  x::AbstractVector)

  ui = x
  vi = op.vi

  lhs!(b,op.odeop,op.ti,(ui,vi),op.ode_cache)

  @. ui = op.u0
  for j = 1:op.i-1
   @. ui = ui  + op.dt * op.a[op.i,j] * op.ki[j]
  end


  q(kk,v) = ∫(kk*v)dΩ
  l(v) = ∫( k*v )dΩ
  kop = AffineFEOperator(q,l,R,W)
  kFE = solve(kop)
  yh = get_free_dof_values(kFE)

  rhs = similar(op.u0)
  DAE_rhs!(rhs,op.odeop,op.ti,(ui,vi),(yh,yh),op.ode_cache)

  @. b = b + rhs
  @. b = -1.0 * b
  b
end

function Gridap.Algebra.jacobian!(A::AbstractMatrix,
  op::DAERKStageOperator,
  x::AbstractVector)
   @. A = op.M
end

import Gridap.ODEs.TransientFETools: allocate_residual


function Gridap.Algebra.allocate_residual(op::DAERKStageOperator,x::AbstractVector)
  Gridap.ODEs.TransientFETools.allocate_residual(op.odeop,op.ti,x,op.ode_cache)
end

function Gridap.Algebra.allocate_jacobian(op::DAERKStageOperator,x::AbstractVector)
  Gridap.ODEs.TransientFETools.allocate_jacobian(op.odeop,op.ti,x,op.ode_cache)
end


function update!(op::DAERKStageOperator,
  ti::Float64,
  ki::AbstractVector,
  i::Int)
  op.ti = ti
  @. op.ki[i] = ki
  op.i = i
end


function get_mass_matrix!(A::AbstractMatrix,
  odeop::ODEOperator,
  t0::Float64,
  u0::AbstractVector,
  ode_cache)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  Gridap.ODEs.ODETools.jacobian!(A,odeop,t0,(u0,u0),2,1.0,ode_cache)
  A
end

""" DIAGNOSTIC FE OPERATOR """



using Gridap.FESpaces
import Gridap.FESpaces: FEOperatorFromWeakForm
import Gridap.FESpaces: get_test
import Gridap.FESpaces: get_trial
import Gridap.FESpaces: collect_cell_vector
import Gridap.Algebra: allocate_vector

""" Allowing for prognostic input """


mutable struct DiagnosticOperator <: NonlinearOperator

end


### FROM FEOPERATORS.jl
import Gridap.FESpaces: AlgebraicOpFromFEOp
function Gridap.FESpaces.allocate_residual(op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  allocate_residual(op.feop,u,F)
end

function Gridap.FESpaces.residual!(b::AbstractVector,op::AlgebraicOpFromFEOp,x::AbstractVector)
  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  residual!(b,op.feop,u,F)
end

function Gridap.FESpaces.residual(op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)
  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  residual(op.feop,u,F)
end

function Gridap.FESpaces.allocate_jacobian(op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)
  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  allocate_jacobian(op.feop,u,F)
end

function Gridap.FESpaces.jacobian!(A::AbstractMatrix,op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)
  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  jacobian!(A,op.feop,u,F)
end

function Gridap.FESpaces.jacobian(op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)
  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  jacobian(op.feop,u,F)
end

function Gridap.FESpaces.residual_and_jacobian!(b::AbstractVector,A::AbstractMatrix,op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)
  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  residual_and_jacobian!(b,A,op.feop,u,F)
end

function Gridap.FESpaces.residual_and_jacobian(op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)
  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  residual_and_jacobian(op.feop,u,F)
end

#### FEOM FEOPERATORSFROMWEAKFORM.jl
mutable struct DiagnosticFEOperator <: FEOperator
  res::Function
  jac::Function
  trial::FESpace
  test::FESpace
  assem::Assembler
end

function DiagnosticFEOperator(
  res::Function,jac::Function,trial::FESpace,test::FESpace)
  assem = SparseMatrixAssembler(trial,test)
  DiagnosticFEOperator(res,jac,trial,test,assem)
end

get_test(feop::DiagnosticFEOperator) = feop.test

get_trial(feop::DiagnosticFEOperator) = feop.trial

get_matrix(feop::DiagnosticFEOperator) = get_matrix(feop.op)

get_vector(feop::DiagnosticFEOperator) = get_vector(feop.op)

get_algebraic_operator(feop::DiagnosticFEOperator) = feop.op

function allocate_residual(op::DiagnosticFEOperator,uh,F)
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(uh,v,F))
  allocate_vector(op.assem, vecdata)
end

function residual!(b::AbstractVector,op::DiagnosticFEOperator,uh,F)
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(uh,v,F))
  assemble_vector!(b,op.assem, vecdata)
  b
end

function allocate_jacobian(op::DiagnosticFEOperator,uh,F)
  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(uh,du,v,F))
  allocate_matrix(op.assem, matdata)
end

function jacobian!(A::AbstractMatrix,op::DiagnosticFEOperator,uh,F)
  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(uh,du,v,F))
  assemble_matrix!(A,op.assem,matdata)
  A
end

function residual_and_jacobian!(
  b::AbstractVector,A::AbstractMatrix,op::DiagnosticFEOperator,uh,F)
  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  data = collect_cell_matrix_and_vector(U,V,op.jac(uh,du,v,F),op.res(uh,v,F))
  assemble_matrix_and_vector!(A, b, op.assem, data)
  (b,A)
end

function residual_and_jacobian(op::DiagnosticFEOperator,uh,F)

  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  data = collect_cell_matrix_and_vector(U,V,op.jac(uh,du,v,F),op.res(uh,v,F))
  A, b = assemble_matrix_and_vector(op.assem, data)
  (b, A)
end



### DIAGNOSTIC SOLVE
res_k(kk,v,F) = ∫(kk*v)dΩ - ∫( k*v*F )dΩ
jac_k(kk,dkk,v,F) = ∫( dkk*v )dΩ
kop = DiagnosticFEOperator(res_k,jac_k,R,W)

kFE = solve(kop)
println(get_free_dof_values(kFE))



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
