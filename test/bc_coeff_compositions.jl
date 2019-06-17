using LinearAlgebra, DiffEqOperators, Random, Test, BandedMatrices

# Generate random parameters
al = rand()
bl = rand()
cl = rand()
dx_l = rand()
ar = rand()
br = rand()
cr = rand()
dx_r = rand()

Q = RobinBC(al, bl, cl, dx_l, ar, br, cr, dx_r)
N = 20
L = CenteredDifference(4,4, 1.0, N)
L2 = CenteredDifference(2,4, 1.0, N)

function coeff_func(du,u,p,t)
  du .= u
end

cL = coeff_func*L
coeffs = rand(N)
DiffEqOperators.update_coefficients!(cL,coeffs,nothing,0.0)

@test cL.coefficients == coeffs

# Fails
u = rand(20)
@test_broken LQ = L*Q
@test_broken LQ*u ≈ L*(Q*u)

u = rand(22)
@test (L + L2) * u ≈ convert(AbstractMatrix,L + L2) * u ≈ (BandedMatrix(L) + BandedMatrix(L2)) * u
