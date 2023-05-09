using JuMP
using Revise
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using SparseDynamicSystem

n = 3
@polyvar x[1:n]
f = [(x[1]^2+x[2]^2-1/4)*x[1], (x[2]^2+x[3]^2-1/4)*x[2], (x[2]^2+x[3]^2-1/4)*x[3]]
g = [1-x[1]^2, 1-x[2]^2, 1-x[3]^2]
d = 3

model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
v, vc, vb = add_poly!(model, x, 2d-2)
w, wc, wb = add_poly!(model, x, 2d)
Lv = v - sum(f .* differentiate(v, x))
model,info1 = add_psatz!(model, Lv, x, g, [], d, QUIET=true, CS=true, TS="block", SO=1, Groebnerbasis=false)
model,info2 = add_psatz!(model, w, x, g, [], d, QUIET=true, CS=true, TS="block", SO=1, Groebnerbasis=false)
model,info3 = add_psatz!(model, w-v-1, x, g, [], d, QUIET=true, CS=true, TS="block", SO=1, Groebnerbasis=false)
supp = get_basis(n, 2d, var=Vector(n:-1:1))
moment = get_moment(n, supp, -ones(n), ones(n))
@objective(model, Min, sum(moment.*wc))
optimize!(model)
status = termination_status(model)
if status != MOI.OPTIMAL
    println("termination status: $status")
    status = primal_status(model)
    println("solution status: $status")
end
objv = objective_value(model)