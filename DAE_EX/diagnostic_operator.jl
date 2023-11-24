struct DiagOperator <: NonlinearOperator
  feop::FEOperator
  end


function Gridap.Algebra.allocate_residual(feop::DiagOperator,u)
  F = interpolate(20,R)
  x = get_free_dof_values(u)
  allocate_residual(feop.op,x,F)
end

function Gridap.Algebra.residual!(b::AbstractVector,feop::DiagOperator,u)
  F = interpolate(20,R)
  x = get_free_dof_values(u)
  residual!(b,feop.op,x,F)
end


function Gridap.Algebra.allocate_jacobian(feop::DiagOperator,u)
  F = interpolate(20,R)
  x = get_free_dof_values(u)
  allocate_jacobian(feop.op,x,F)
end

function Gridap.Algebra.jacobian!(A::AbstractMatrix,feop::DiagOperator,u)
  F = interpolate(20,R)
  x = get_free_dof_values(u)
  jacobian!(A,feop.op,x,F)
end
