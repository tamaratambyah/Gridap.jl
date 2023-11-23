using Pkg

using Gridap
using Gridap.Algebra, Gridap.ODEs.ODETools, Gridap.ODEs.TransientFETools
using LinearAlgebra

# using Plots
# using Test
# using GridapSolvers

# initial condition
k = 1.0

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



uh0 = interpolate_everywhere(u(0.0),U(0.0))
u0 = get_free_dof_values(uh0)
K = interpolate_everywhere(k,U(0.0))


function lhs(t,u,v)
  return ∫( u*v )dΩ
end
function rhs(t,u,v,k)
  return ∫(v*f(t))dΩ - ∫(( k*∇(v)⊙∇(u) ))dΩ
end
# function rhs(t,u,v)
#   return  ∫( u*v )dΩ
# end



## FE OPERATOR

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

# get_order(op)
import Gridap.ODEs.TransientFETools: rhs!
function Gridap.ODEs.TransientFETools.rhs!(
  rhs::AbstractVector,
  op::DAE,
  t::Real,
  xh::T,
  # yh::T,
  cache) where T

  yh = interpolate_everywhere(k,U(t))

  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.rhs(t,xh,v,yh))
  assemble_vector!(rhs,op.assem_t,vecdata)
  rhs
end


import Gridap.ODEs.TransientFETools: residual!
function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::DAE,
  t::Real,
  xh::T,
  cache) where T
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(t,xh,v))
  assemble_vector!(b,op.assem_t,vecdata)
  b
end

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

op = EXRKDAE(lhs,rhs,U,V)


# _rhs = similar(u0)
# rhs!(_rhs,op,0.0,uh0,F,nothing)

# _b = similar(u0)
# lhs!(_b,op,0.0,uh0,nothing)

# _b = similar(u0)
# residual!(_b,op,0.0,uh0,nothing)

# allocate_residual(op,0.0,uh0,F,nothing)


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

# A = allocate_jacobian(op,0.0,uh0,nothing)
# jacobian!(A,op,0.0,uh0,1,1,nothing)

# uh = uh0
# dxh = ()
#   for i in 1:get_order(op)
#     dxh = (dxh...,uh)
#   end
# xh = TransientCellField(uh,dxh)
# jacobians!(A,op,0.0,xh,(1,1),nothing)

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


  rhs = similar(op.u0)
  rhs!(rhs,op.odeop,op.ti,(ui,vi),op.ode_cache)

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

import Gridap.FESpaces: get_algebraic_operator
import Gridap.ODEs.ODETools: allocate_cache

a = reshape([0.0],1,1)
b = [1.0]
c = [0.0]

odeop = get_algebraic_operator(op)
t0 = 0.0
dt = 0.001
ode_cache = allocate_cache(odeop)
vi = similar(u0)

ki = [similar(u0)]
M = allocate_jacobian(op,0.0,uh0,nothing)
get_mass_matrix!(M,odeop,t0,u0,ode_cache)

lop = DAERKStageOperator(odeop,t0,dt,u0,ode_cache,vi,ki,0,a,M)

uf = similar(u0)
ls = LUSolver()
l_cache = nothing



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
u0 = copy(u_ex)
t0 = tf
ki = [similar(u0)]
M = allocate_jacobian(op,0.0,uh0,nothing)
get_mass_matrix!(M,odeop,t0,u0,ode_cache)

lop = DAERKStageOperator(odeop,t0,dt,u0,ode_cache,vi,ki,0,a,M)

uf = similar(u0)
ls = LUSolver()
l_cache = nothing
