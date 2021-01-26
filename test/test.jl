using DynamicPolynomials

# The Lorentz system
@polyvar x[1:5]
ρ = 28
σ = 10
β = 8/3
f = [σ * (x[2] - x[1]), x[1] * (ρ - x[3]) - x[2], x[1] * x[2] - β*x[3], σ * (x[4] - x[1]), x[1] * (ρ - x[3]) - x[5]]
# g = [10 - ρ*x[1]^2 -  σ*x[2]^2 -  σ*(x[3] - 2*ρ)^2]
g = [1-x[1]^2, 1-x[2]^2, 1-x[3]^2, 1-x[4]^2, 1-x[5]^2]
@time begin
opt,w=MPI_first(f, g, x, 4, -ones(5), ones(5), β=1, TS="block")
end

# the Van-der-Pol oscillator
@polyvar x[1:2]
f = [2*x[2], -0.8*x[1] - 10*(x[1]^2-0.21)*x[2]]
g = [1.1^2-x[1]^2, 1.1^2-x[2]^2]
@time begin
opt,w=MPI_first(f, g, x, 8, -1.1*ones(2), 1.1*ones(2), β=1, TS="block")
end

@polyvar x[1:6]
f = [2*x[2], -0.8*x[1] - 10*(x[1]^2-0.21)*x[2] + 0.1*x[5], 2*x[4], -0.8*x[3] - 10*(x[3]^2-0.21)*x[4] + 0.1*x[1], 2*x[6], -0.8*x[5] - 10*(x[5]^2-0.21)*x[6] + 0.1*x[3]]
g = [1.1^2-x[1]^2, 1.1^2-x[2]^2, 1.1^2-x[3]^2, 1.1^2-x[4]^2, 1.1^2-x[5]^2, 1.1^2-x[6]^2]
opt,w=MPI_first(f, g, x, 5, -1.1*ones(6), 1.1*ones(6), TS="block")
