### Tranisent Jacobians - allowing for op::DAE


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
