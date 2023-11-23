using Pkg

using Gridap
using Gridap.Algebra, Gridap.ODEs.ODETools
using LinearAlgebra

# using Plots
# using Test
# using GridapSolvers

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
g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
U = TransientTrialFESpace(V,g)

# initial condition

u(x,t) = x[1]*(1-x[1])*t
u(t) = x -> u(x,t)
uh0 = interpolate_everywhere(u(0.0),U(0.0))


function rhs(t,u,v,F)
  return  ∫( u*v*F )dΩ
end



#### testing rhs assembly - single field

using Gridap.FESpaces
import Gridap.FESpaces: collect_cell_vector
import Gridap.FESpaces: assemble_vector
import Gridap.FESpaces: get_fe_basis

u0 = get_free_dof_values(uh0)

_u = uh0
_F = (interpolate_everywhere(1.0,U(0.0)))

function my_rhs!(
  rhs_vec,
  t::Real,
  xh,
  yh,
  test,
  trial,
  rhs_func
  )
  v = get_fe_basis(trial)
  vecdata = collect_cell_vector(trial,rhs_func(t,xh,v,yh))
  assem_t = SparseMatrixAssembler(test,trial)
  assemble_vector!(rhs_vec,assem_t,vecdata)
  rhs_vec
end
rhs_vec = similar(u0)
my_rhs!(rhs_vec,0.0,_u,_F,U,V,rhs)


#### testing rhs assembly - multi field
function rhs_multi(t,(u1,u2),(v1,v2),F)
  return  ∫( u1*v1*F + u2*v2*F )dΩ
end

X = TransientMultiFieldFESpace([U,U])
Y = MultiFieldFESpace([V,V])
xh0 = interpolate_everywhere([uh0,uh0],X(0.0))

x0 = get_free_dof_values(xh0)
_u_multi = xh0
rhs_multi_vec = similar(x0)
my_rhs!(rhs_multi_vec,0.0,_u_multi,_F,X,Y,rhs_multi)
