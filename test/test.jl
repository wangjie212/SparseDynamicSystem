# include("E:\\Programs\\SparseDynamicSystem\\SparseDynamicSystem\\src\\SparseDynamicSystem.jl")
using SparseDynamicSystem
using DynamicPolynomials
# using Plots
using LinearAlgebra
using LightGraphs
using DelimitedFiles

# The Lorentz system
@polyvar x[1:5]
f = [10*x[1]-12*x[2], -70/3*x[1]+x[2]+125/3*x[1]*x[3], 8/3*x[3]-15*x[1]*x[2], 10*(x[4]-x[1]), x[1]*(28-x[3])-x[5]]
g = [1-x[1]^2, 1-x[2]^2, 1-x[3]^2, 1-x[4]^2, 1-x[5]^2]
@time begin
opt,w = MPI(f, g, x, 5, -ones(5), ones(5), QUIET=false, merge=true, md=3, TS=["block","MD"], SO=[1;1], β=1)
end

n = 6
B = zeros(Float16, n, n)
while eigmin(B) <= 0
    G = erdos_renyi(n, n-4)
    A = adjacency_matrix(G)
    for j = 1:n, k = j:n
        if j == k
            B[j,k] = rand(1)[1]+1
        elseif A[j,k] == 1
            B[j,k] = rand(1)[1]-0.5
            B[k,j] = B[j,k]
        else
            B[k,j],B[j,k] = 0,0
        end
    end
end

@polyvar x[1:n]
f = -Diagonal(ones(n))*x + x'*B*x*x
g = [1-x[i]^2 for i=1:n]
@time begin
opt,w = MPI(f, g, x, 5, -ones(n), ones(n), QUIET=false, merge=true, md=3, TS=["block","block"], SO=[2;1], β=1)
end
@time begin
opt,w = MPI(f, g, x, 3, -ones(n), ones(n), QUIET=false, merge=true, md=3, TS=[false,false], SO=[1;1], β=1)
end
@time begin
opt,w = MPI(f, g, x, 6, -ones(n), ones(n), QUIET=false, merge=true, md=2, TS=["block","block"], SO=[1;2], β=1)
end

io = open("E:\\Programs\\SparseDynamicSystem\\data\\random_10.txt", "w")
writedlm(io, B)
close(io)
io = open("E:\\Programs\\SparseDynamicSystem\\data\\random_6.txt", "r")
B = readdlm(io, Float16)
close(io)

# the Van-der-Pol oscillator
@polyvar x[1:2]
f = [2*x[2], -0.8*x[1] - 10*(x[1]^2-0.21)*x[2]]
g = [1.1^2-x[1]^2, 1.1^2-x[2]^2]
@time begin
opt,w = MPI(f, g, x, 10, -1.1*ones(2), 1.1*ones(2), TS=["block","block"], SO=[3;1], β=1)
end

@polyvar x[1:6]
f = [2*x[2], -0.8*x[1] - 10*(x[1]^2-0.21)*x[2] + 0.1*x[5], 2*x[4], -0.8*x[3] - 10*(x[3]^2-0.21)*x[4] + 0.1*x[1], 2*x[6], -0.8*x[5] - 10*(x[5]^2-0.21)*x[6] + 0.1*x[3]]
g = [1-x[1]^2, 1-x[2]^2, 1-x[3]^2, 1-x[4]^2, 1-x[5]^2, 1-x[6]^2]
opt,w = MPI(f, g, x, 5, -ones(6), ones(6), TS=["block", "block"], SO=[2;1])

# 9 mode fluid model
α = 1/2
β = π/2
κ1 = sqrt(α^2 + 1)
κ2 = sqrt(β^2 + 1)
κ3 = sqrt(α^2 + β^2 +1)
R = 100
@polyvar a[1:9]
f = Vector{Polynomial}(undef, 9)
f[1] = β^2/R - β^2/R*a[1] - sqrt(3/2)*β/κ3*a[6]*a[8] + sqrt(3/2)*β/κ2*a[2]*a[3]
f[2] = -(4*β^2/3 + 1)*a[2]/R + 5*sqrt(2)*a[4]*a[6]/(3*sqrt(3)*κ1) - a[5]*a[7]/(sqrt(6)*κ1)
- α*β*a[5]*a[8]/(sqrt(6)*κ1*κ3) - sqrt(3/2)*β/κ2*a[1]*a[3] - sqrt(3/2)*β/κ2*a[3]*a[9]
f[3] = -κ2^2/R*a[3] + 2*α*β*(a[4]*a[7] + a[5]*a[6])/(sqrt(6)*κ1*κ2) + (β^2*(3*α^2 + 1)
- 3*κ1^2)*a[4]*a[8]/(sqrt(6)*κ1*κ2*κ3)
f[4] = -(3*α^2 + 4*β^2)/(3*R)*a[4] - α/sqrt(6)*a[1]*a[5] - (10*α^2*a[2]*a[6])/(3sqrt(6)*κ1)
- sqrt(3/2)*α*β/(κ1*κ2)*a[3]*a[7] - sqrt(3/2)*α^2*β^2/(κ1*κ2*κ3)*a[3]*a[8] - α/sqrt(6)*a[5]*a[9]
f[5] = -(α^2+β^2)/R*a[5] + α/sqrt(6)*a[1]*a[4] + α^2/(sqrt(6)*κ1)*a[2]*a[7] - α*β/(sqrt(6)*κ1*κ3)*a[2]*a[8]
+ 2*α*β/(sqrt(6)*κ1*κ2)*a[3]*a[6] + α/sqrt(6)*a[4]*a[9]
f[6] = -(3*α^2 + 4*β^2 + 3)/(3*R)*a[6] + α/sqrt(6)*a[1]*a[7] + sqrt(3/2)*β/κ3*a[1]*a[8] + (10*α^2 - 1)/(3*sqrt(6)*κ1)*a[2]*a[4]
- 2*α*β*sqrt(2/3)/(κ1*κ2)*a[3]*a[5] + α/sqrt(6)*a[7]*a[9] + sqrt(3/2)*β/κ3*a[8]*a[9]
f[7] = -κ3^2/R*a[7] - α/sqrt(6)*(a[1]*a[6] + a[6]*a[9]) - (α^2 - 1)/(sqrt(6)*κ1)*a[2]*a[5]
+ α*β/(sqrt(6)*κ1*κ2)*a[3]*a[4]
f[8] = -κ3^2/R*a[8] + 2*α*β/(sqrt(6)*κ1*κ3)*a[2]*a[5] + (3*α^2 - β^2 + 3)/(sqrt(6)*κ1*κ2*κ3)*a[3]*a[4]
f[9] = -9*β^2/R*a[9] + sqrt(3/2)*β/κ2*a[2]*a[3] - sqrt(3/2)*β/κ3*a[6]*a[8]
g = [1 - sum(a.^2)]
h = -(β^2*a[1]^2 + (4*β^2/3 + 1)*a[2]^2 + κ2^2*a[3]^2 + (3*α^2 + 4*β^2)/3*a[4]^2 + (α^2+β^2)*a[5]^2 +
(3*α^2 + 4*β^2 + 3)/3*a[6]^2 + κ3^2*a[7]^2 + κ3^2*a[8]^2 + 9*β^2*a[9]^2)
# h = (1-a[1])^2 + sum(a[2:9].^2)

@time begin
opt = UPO(f, h, g, a, 3, TS="block", SO=2)
end

# Powergrid 1
@polyvar x[1:6]
f = Vector{Polynomial}(undef, 6)
f[1] = 0.4996*x[4] - 0.4*x[1] - 1.4994*x[3] - 0.02*x[6] + 0.02*x[3]*x[4] + 0.4996*x[3]*x[6] - 0.4996*x[4]*x[5] + 0.02*x[5]*x[6]
f[2] = 0.4996*x[3] + 0.02*x[5] - 0.9986*x[4] + 0.05*x[6] - 0.5*x[2] - 0.02*x[3]*x[4] - 0.4996*x[3]*x[6] + 0.4996*x[4]*x[5] - 0.02*x[5]*x[6]
f[3] = (1 - x[5])*x[1]
f[4] = (1 - x[6])*x[2]
f[5] = x[1]*x[3]
f[6] = x[2]*x[4]
p = [1-x[1]^2, 1-x[2]^2, 1-x[3]^2, 1-x[4]^2, 1-x[5]^2, 1-x[6]^2]
q = [0.01-sum(x.^2)]

@time begin
opt,w = MPI(f, p, x, 5, -ones(6), ones(6), β=1, SO=[1,1], merge=true, md=4, TS=["MD", "MD"])
end

@time begin
opt,v0 = ROA(f, p, q, x, 8, 3, -ones(6), ones(6), SO=[1,1], merge=true, md=4, TS=["block","block"])
end

@polyvar x[1:6]
f = Vector{Polynomial}(undef, 6)
f[1] = - x[3] - 0.5x[3]*x[6] + 0.5x[4]*x[5] - 0.4x[1]
f[2] = - 0.5x[4] + 0.5x[3]*x[6] - 0.5x[4]*x[5] - 0.5x[2] + 0.05
f[3] = x[1]*x[5]
f[4] = x[2]*x[6]
f[5] = - x[1]*x[3]
f[6] = - x[2]*x[4]
p = [1-x[1]^2, 1-x[2]^2, 1-x[3]^2, 1-x[4]^2, 1-x[5]^2, 1-x[6]^2]
q = [0.01-sum(x.^2)]

# 16 mode fluid model
@polyvar a[1:16]
f = Vector{Polynomial}(undef, 16)
f[1] = -(2*π)^2*a[1] + sqrt(2)*π*sum(a[j]*a[j+1] for j=1:15)
for i = 2:15
    f[i] = -(2*π*i)^2*a[i] + sqrt(2)*π*i*(sum(a[j]*a[j+i] for j=1:16-i) - 1/2*sum(a[j]*a[i-j] for j=1:i-1))
end
f[16] = -(2*π*16)^2*a[16] - sqrt(2)*π*8*sum(a[j]*a[16-j] for j=1:15)
h = 2*π^2*sum(i^2*a[i]^2 for i=1:16)

t = zeros(10)
val = zeros(10)
Φ0 = 10^3
for i = 1:10
Φ = i*Φ0
g = [h/Φ^(1/4) - Φ^(3/4)]
o = [Φ^(3/4)/(2*π^2) - sum(a.^2)/Φ^(1/4)]

t[i] = @elapsed begin
opt = BEE(f, h, o, g, a, 2, TS=["MD","MD"], SO=[1,1], merge=true, md=4, QUIET=false)
end
val[i] = opt/Φ^(3/2)
end

cp = subs(w, x[3:n] => zeros(n-2))
a = Float16[]
b = Float16[]
for i=-1:0.002:1, j=-1:0.002:1
    if  polynomial(cp)(x[1:2] => [i;j]) >= 1
        push!(a, i)
        push!(b, j)
    end
end
plot(a, b, seriestype=:scatter, xlims=[-1;1], ylims=[-1;1], markersize=0.1, markercolor="blue", label=["TS, 2d=4"])
savefig("E:\\Programs\\SparseDynamicSystem\\data\\random_12_2_TS.png")
