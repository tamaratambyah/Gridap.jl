
abstract type DAEFEOperator <: FEOperator end

## F = prognostic variable
## u = diagnostic variable

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

import Gridap.FESpaces: get_algebraic_operator
function get_algebraic_operator(op::DAEFEOperator)
  op.feop
end

# """
# AlgebraicOpFromFEOp
# """
import Gridap.FESpaces: AlgebraicOpFromFEOp
struct AlgebraicDAEOpFromFEOp <: NonlinearOperator
  feop::FEOperator
  f::AbstractVector  # dofs of prognostic variable (known)
  space::FESpace
end

function Gridap.FESpaces.allocate_residual(op::AlgebraicDAEOpFromFEOp,x::AbstractVector)
  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.allocate_residual(op.feop,u,F)
end

function Gridap.FESpaces.residual!(b::AbstractVector,op::AlgebraicDAEOpFromFEOp,x::AbstractVector)
  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.residual!(b,op.feop,u,F)
end

function Gridap.FESpaces.residual(op::AlgebraicDAEOpFromFEOp,x::AbstractVector)
  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.residual(op.feop,u,F)
end

function Gridap.FESpaces.allocate_jacobian(op::AlgebraicDAEOpFromFEOp,x::AbstractVector)
  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.allocate_jacobian(op.feop,u,F)
end

function Gridap.FESpaces.jacobian!(A::AbstractMatrix,op::AlgebraicDAEOpFromFEOp,x::AbstractVector)
  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.jacobian!(A,op.feop,u,F)
end

function Gridap.FESpaces.jacobian(op::AlgebraicDAEOpFromFEOp,x::AbstractVector)
  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.jacobian(op.feop,u,F)
end

function Gridap.FESpaces.residual_and_jacobian!(b::AbstractVector,A::AbstractMatrix,
  op::AlgebraicDAEOpFromFEOp,x::AbstractVector)
  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.residual_and_jacobian!(b,A,op.feop,u,F)
end

function Gridap.FESpaces.residual_and_jacobian(op::AlgebraicDAEOpFromFEOp,x::AbstractVector)

  f = op.f
  space = op.space

  F = EvaluationFunction(space,f)

  trial = get_trial(op.feop)
  u = EvaluationFunction(trial,x)
  Gridap.FESpaces.residual_and_jacobian(op.feop,u,F)
end

function Gridap.FESpaces.zero_initial_guess(op::AlgebraicDAEOpFromFEOp)
  trial = get_trial(op.feop)
  x = zero_free_values(trial)
end


# function update!(op::AlgebraicDAEOpFromFEOp,
#   ti::Float64,
#   u::AbstractVector,
#   f::AbstractVector)

#   op.ti = ti
#   op.u = u
#   op.f = f
# end
