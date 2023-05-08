using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using SparseDynamicSystem

@polyvar y[1:3]
x = reverse(y)
f = [(x[1]^2+x[2]^2-1/4)*x[1], (x[2]^2+x[3]^2-1/4)*x[2], (x[2]^2+x[3]^2-1/4)*x[3]]
g = [1-x[1]^2, 1-x[2]^2, 1-x[3]^2]
d = 3

model = Model(optimizer_with_attributes(Mosek.Optimizer))
set_optimizer_attribute(model, MOI.Silent(), true)
vb = reverse(monomials(y, 0:2d-2))
vc = @variable(model, [1:length(vb)])
v = vc'*vb
wb = reverse(monomials(y, 0:2d))
wc = @variable(model, [1:length(wb)])
w = wc'*wb
tsupp = get_basis(3, 2d)
moment = get_moment(3, tsupp, -ones(3), ones(3))
Lv = v - sum(f .* differentiate(v, x))
model,_,_,_,_,_,_ = add_psatz!(model, Lv, x, g, [], d, QUIET=true, CS=true, TS="block", SO=1, Groebnerbasis=false)
model,_,_,_,_,_,_ = add_psatz!(model, w, x, g, [], d, QUIET=true, CS=true, TS="block", SO=1, Groebnerbasis=false)
model,_,_,_,_,_,_ = add_psatz!(model, w-v-1, x, g, [], d, QUIET=true, CS=true, TS="block", SO=1, Groebnerbasis=false)
@objective(model, Min, sum(moment.*wc))
optimize!(model)
status = termination_status(model)
if status != MOI.OPTIMAL
    println("termination status: $status")
    status = primal_status(model)
    println("solution status: $status")
end
objv = objective_value(model)