## Stage operator to be used in RK

using Gridap
using Gridap.Algebra, Gridap.ODEs.ODETools, Gridap.ODEs.TransientFETools
using LinearAlgebra

using Gridap.FESpaces
import Gridap.FESpaces: collect_cell_vector
import Gridap.FESpaces: assemble_vector
import Gridap.FESpaces: get_fe_basis

import Gridap.ODEs.TransientFETools: TransientFEOperator
import Gridap.ODEs.TransientFETools: OperatorType
import Gridap.ODEs.TransientFETools: get_test
import Gridap.ODEs.TransientFETools: TransientCellField


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
