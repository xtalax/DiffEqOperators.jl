using Test, DiffEqOperators
using OrdinaryDiffEq

@testset "Parabolic Heat Equation with Dirichlet BCs" begin
    x = collect(-pi : 2pi/511 : pi)
    u0 = @. -(x - 0.5).^2 + 1/12
    Dxx = CenteredDifference(2,2,2π/511,length(x))
    Qx = DirichletBC(u0[1], u0[end])
    Lxx = Dxx*Qx
    heat_eqn = ODEProblem(Lxx, u0[2:end-1], (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)
    # Broken in different amounts on each CI computer
    for t in 0:0.1:10
        @test soln(t)[1] ≈ u0[1]
        @test soln(t)[end] ≈ u0[end]
    end
end

@testset "Parabolic Heat Equation with Neumann BCs" begin
    N = 512
    dx = 2π/(N-1)
    x = collect(-pi : dx : pi)
    u0 = @. -(x - 0.5)^2 + 1/12
    B = CenteredDifference(1,2,dx,N)
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]

    A = CenteredDifference(2,2,dx,N)
    Q = NeumannBC([deriv_start, deriv_end], dx, 3)

    heat_eqn = ODEProblem(A*Q, u0[2:end-1], (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (1/dx)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (1/dx))

    for t in 0:0.1:10
        @test sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ deriv_start atol=1e-1
        @test sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ deriv_end atol=1e-1
    end
end

@testset "Parabolic Heat Equation with Robin BCs" begin
    N = 512
    dx = 2π/(N-1)
    x = collect(-pi : dx : pi)
    u0 = @. -(x - 0.5).^2 + 1/12
    B = CenteredDifference(1,2,dx,N,:None,:None)
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]
    params = [1.0,0.5]

    left_RBC = params[1]*u0[1] - params[2]*deriv_start
    right_RBC = params[1]*u0[end] + params[2]*deriv_end

    A = CenteredDifference(2,2,dx,N,)
    Q = RobinBC([params..., left_RBC], [params..., right_RBC] ,dx, 3)
    heat_eqn = ODEProblem(A*Q, u0[2:end-1], (0.,10.));
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10);

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (1/dx)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (1/dx))
    val = []
    # Broken in different amounts on each CI computer
    for t in 0.2:0.1:9.8
        @test params[1]*soln(t)[1] + -params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ left_RBC atol=1e-1
        # append!(val,params[1]*soln(t)[1] + -params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) - left_RBC)
        @test params[1]*soln(t)[end] + params[2]*sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ right_RBC atol=1e-1
    end
end



function heat_sol(space::NTuple{2,AbstractVector(T)}, coeffs::NTuple{2,AbstractVector{T}}; α = 1.0) where {T}
    L = map(x -> x[end]-x[1], space)
    ω₀ = π./L
    u = zeros(T,length.(space))
    function _heat_sol(t::T1) where T1
        for (j,x) in enumerate(space[2]), (i,x) in enumerate(space[1])
            for (ny, cy) in enumerate(coeffs[2]), (nx, cx) in enumerate(coeffs[1])
                n = (nx,ny)
                ωx, ωy = (n.-1).*ω₀
                u[i,j] += cx*cy*cos(ωx*x)*cos(ωy*y)*exp(-α*(ωx^2 + ωy^2) *t)
            end
        end
        return u
    end
    return _heat_sol
end


#Tests a solution of the heat equation:
# ∂ₜu = α⋅∇²u
# with neumann 0 boundary conditions and some 2d harmonic cosine superposition as initial condition.
#
# Using a frequency domain approach, an analyitic solution to this equation is found:
# u(t,x,y) = ∑{n∈N,m∈M}[Cₙₘ⋅cos(nω₀x)⋅cos(mω₀y)*exp(-((nω₀)^2+(mω₀)^2)⋅α⋅t)]
# The numerical finite difference approach is tested against this solution.
#
@testset "Harmonic 2D Heat Equation with Robin BCs" begin
N = 25 #N of harmonnics in each dimension
ωmax = (N-1)/2.0
Δx = 1/(ωmax*10.0) #Ensure that the spatial resolution is sufficiently far above the nyquist sampling criteria for this function (f_sampling > 2*fmax)
x = collect(-π : Δx : π)
y = collect(-π : Δx : π)
c̄ = (rand(N), rand(N)) # random fourier coefficients
analytic_soln = heat_sol((x,y), 0.0, c; α = 1.0)

u0 = analytic_soln(0.0)
u_interior = u0[2:end-1, 2:end-1] #slice out the interior of u, leaving off the boundary points

#set up time
Δt = 0.05
T = Δt:Δt:10.0
#set up operators
∇² = CenteredDifference{1}(2, 4, Δx, length(x) + CenteredDifference{2}(2, 4, Δx, length(y)#that is one gorgeous laplacian right there
Q = compose(Neumann0BC(Float64, (Δx,Δx), 5, size(u_interior))...) #Neumann everywhere, no energy enters or leaves the region

L = ∇²*Q # Create full Operator with boundary conditions

heat_eqn = ODEProblem(L, u0[2:end-1], (0.,10.));
soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10);

#U = fill(zeros(size(u0)), length(T))
#use a time stepping solve -
for (i,t) in enumerate(0:0.01:10)
    uᵢ .= uᵢ .+ L*uᵢ
    #U[i] = uᵢ
    @test soln(t) ≈ analytic_soln(t)[2:end-1,2:end-1] atol = 0.01;
end
#= run these for a nice animation
using Plots
pyplot()
@gif for u in U
    heatmap(x,y,u)
end
=#
