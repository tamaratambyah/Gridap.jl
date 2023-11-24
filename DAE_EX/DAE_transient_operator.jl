### Allow for rhs to have diagnosic input


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
