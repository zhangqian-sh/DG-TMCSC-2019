using LinearAlgebra
using QuadGK
using DelimitedFiles

# parameters
h = 1 / 10
N = Int(1 / h)

# basis function
ϕ⁰(x, xⱼ) = 1
ϕ¹(x, xⱼ) = (x - xⱼ) / 0.5h
ϕ²(x, xⱼ) = 0.5(3((x - xⱼ) / 0.5h)^2 - 1)

# LHS matrix
A = [1 1 1; -1 1 1; 1 -1 1]

# equation
f(x) = cos(x)

# boundary condition
a = 0 # left boundary is 0

# DG scheme
function solve_eqn(a)
    u = zeros(Float64, (N, 3))
    for j = 1:N
        xⱼ = (j - 1) * h + 0.5h
        I₀, _ = quadgk(x -> f(x) * ϕ⁰(x, xⱼ), xⱼ - 0.5h, xⱼ + 0.5h)
        I₁, _ = quadgk(x -> f(x) * ϕ¹(x, xⱼ), xⱼ - 0.5h, xⱼ + 0.5h)
        I₂, _ = quadgk(x -> f(x) * ϕ²(x, xⱼ), xⱼ - 0.5h, xⱼ + 0.5h)
        b = [a + I₀; -a + I₁; a + I₂] #]
        u[j, :] = A \ b
        a = u[j, 1] * ϕ⁰(xⱼ + 0.5h, xⱼ) + u[j, 2] * ϕ¹(xⱼ + 0.5h, xⱼ) + u[j, 3] * ϕ²(xⱼ + 0.5h, xⱼ)
    end
    u
end

u = solve_eqn(a)

# evaluate at central points (debug)
y = zeros(Float64, (N))
y_true = zeros(Float64, (N))
for j = 1:N
    xⱼ = (j - 1) * h + 0.5h
    y[j] = u[j, 1] * ϕ⁰(xⱼ, xⱼ) + u[j, 2] * ϕ¹(xⱼ, xⱼ) + u[j, 3] * ϕ²(xⱼ, xⱼ)
    y_true[j] = sin(xⱼ)
end
println(y)
println(y_true)

# save results to file
writedlm("result.csv", u, ",")
