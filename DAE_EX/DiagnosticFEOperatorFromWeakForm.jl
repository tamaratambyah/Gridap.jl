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

struct DiagnosticFEOperatorFromWeakForm <: FEOperator
  res::Function
  jac::Function
  trial::FESpace
  test::FESpace
  assem::Assembler
end

function DiagnosticFEOperator(
  res::Function,jac::Function,trial::FESpace,test::FESpace)
  assem = SparseMatrixAssembler(trial,test)
  DiagnosticFEOperatorFromWeakForm(res,jac,trial,test,assem)
end

Gridap.FESpaces.get_test(op::DiagnosticFEOperatorFromWeakForm) = op.test

Gridap.FESpaces.get_trial(op::DiagnosticFEOperatorFromWeakForm) = op.trial



function Gridap.FESpaces.allocate_residual(op::DiagnosticFEOperatorFromWeakForm,uh,F)
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(uh,v,F))
  allocate_vector(op.assem, vecdata)
end

function Gridap.FESpaces.residual!(b::AbstractVector,op::DiagnosticFEOperatorFromWeakForm,uh,F)
  V = get_test(op)
  v = get_fe_basis(V)
  vecdata = collect_cell_vector(V,op.res(uh,v,F))
  assemble_vector!(b,op.assem, vecdata)
  b
end

function Gridap.FESpaces.allocate_jacobian(op::DiagnosticFEOperatorFromWeakForm,uh,F)
  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(uh,du,v,F))
  allocate_matrix(op.assem, matdata)
end

function Gridap.FESpaces.jacobian!(A::AbstractMatrix,op::DiagnosticFEOperatorFromWeakForm,uh,F)
  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(U,V,op.jac(uh,du,v,F))
  assemble_matrix!(A,op.assem,matdata)
  A
end

function Gridap.FESpaces.residual_and_jacobian!(
  b::AbstractVector,A::AbstractMatrix,op::DiagnosticFEOperatorFromWeakForm,uh,F)
  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  data = collect_cell_matrix_and_vector(U,V,op.jac(uh,du,v,F),op.res(uh,v,F))
  assemble_matrix_and_vector!(A, b, op.assem, data)
  (b,A)
end

function Gridap.FESpaces.residual_and_jacobian(op::DiagnosticFEOperatorFromWeakForm,uh,F)

  U = get_trial(op)
  V = get_test(op)
  du = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  data = collect_cell_matrix_and_vector(U,V,op.jac(uh,du,v,F),op.res(uh,v,F))
  A, b = assemble_matrix_and_vector(op.assem, data)
  (b, A)
end
