
abstract type DAEFEOperator <: FEOperator end


function Gridap.FESpaces.residual(op::DAEFEOperator,u,F)
  b = allocate_residual(op,u,F)
  residual!(b,op,u,F)
  b
end


function Gridap.FESpaces.jacobian(op::DAEFEOperator,u,F)
  A = allocate_jacobian(op,u,F)
  jacobian!(A,op,u,F)
  A
end


function Gridap.FESpaces.residual_and_jacobian!(b::AbstractVector,A::AbstractMatrix,op::DAEFEOperator,u,F)
  residual!(b,op,u,F)
  jacobian!(A,op,u,F)
  (b,A)
end

function Gridap.FESpaces.residual_and_jacobian(op::DAEFEOperator,u,F)
  b = residual(op,u,F)
  A = jacobian(op,u,F)
  (b,A)
end

"""
AlgebraicOpFromFEOp
"""

import Gridap.FESpaces: AlgebraicOpFromFEOp
function Gridap.FESpaces.allocate_residual(op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  allocate_residual(op.feop,u,F)
end

function Gridap.FESpaces.residual!(b::AbstractVector,op::AlgebraicOpFromFEOp,x::AbstractVector)
  F = interpolate(20,R)

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

function Gridap.FESpaces.zero_initial_guess(op::AlgebraicOpFromFEOp)
  trial = get_trial(op.feop)
  x = zero_free_values(trial)
end
