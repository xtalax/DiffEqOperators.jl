# ~ bound checking functions ~
checkbounds(A::AbstractDerivativeOperator, k::Integer, j::Integer) =
    (0 < k ≤ size(A, 1) && 0 < j ≤ size(A, 2) || throw(BoundsError(A, (k,j))))

checkbounds(A::AbstractDerivativeOperator, kr::AbstractRange, j::Integer) =
    (checkbounds(A, first(kr), j); checkbounds(A,  last(kr), j))

checkbounds(A::AbstractDerivativeOperator, k::Integer, jr::AbstractRange) =
    (checkbounds(A, k, first(jr)); checkbounds(A, k,  last(jr)))

checkbounds(A::AbstractDerivativeOperator, kr::AbstractRange, jr::AbstractRange) =
    (checkbounds(A, kr, first(jr)); checkbounds(A, kr,  last(jr)))

checkbounds(A::AbstractDerivativeOperator, k::Colon, j::Integer) =
    (0 < j ≤ size(A, 2) || throw(BoundsError(A, (size(A,1),j))))

checkbounds(A::AbstractDerivativeOperator, k::Integer, j::Colon) =
    (0 < k ≤ size(A, 1) || throw(BoundsError(A, (k,size(A,2)))))

@inline function getindex(A::AbstractDerivativeOperator, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    bpc = A.boundary_point_count
    N = A.len
    bsl = A.boundary_stencil_length
    slen = A.stencil_length
    if bpc > 0 && 1<=i<=bpc
        if j > bsl
            return 0
        else
            return A.low_boundary_coefs[i][j]
        end
    elseif bpc > 0 && (N-bpc)<i<=N
        if j < N+2-bsl
            return 0
        else
            return A.high_boundary_coefs[bpc-(N-i)][bsl-(N+2-j)]
        end
    else
        if j < i-bpc || j > i+slen-bpc-1
            return 0
        else
            return A.stencil_coefs[j-i + 1 + bpc]
        end
    end
end

# scalar - colon - colon
@inline getindex(A::AbstractDerivativeOperator, ::Colon, ::Colon) = Array(A)

@inline function getindex(A::AbstractDerivativeOperator, ::Colon, j)
    return BandedMatrix(A)[:,j]
end


# symmetric right now
@inline function getindex(A::AbstractDerivativeOperator, i, ::Colon)
    return BandedMatrix(A)[i,:]
end


# UnitRanges
@inline function getindex(A::AbstractDerivativeOperator, rng::UnitRange{Int}, ::Colon)
    m = BandedMatrix(A)
    return m[rng, cc]
end


@inline function getindex(A::AbstractDerivativeOperator, ::Colon, rng::UnitRange{Int})
    m = BandedMatrix(A)
    return m[rnd, cc]
end

@inline function getindex(A::AbstractDerivativeOperator, r::Int, rng::UnitRange{Int})
    m = A[r, :]
    return m[rng]
end


@inline function getindex(A::AbstractDerivativeOperator, rng::UnitRange{Int}, c::Int)
    m = A[:, c]
    return m[rng]
end


@inline function getindex(A::AbstractDerivativeOperator{T}, rng::UnitRange{Int}, cng::UnitRange{Int}) where T
    return BandedMatrix(A)[rng,cng]
end

#=
    This definition of the mul! function makes it possible to apply the LinearOperator on
    a matrix and not just a vector. It basically transforms the rows one at a time.
=#
function LinearAlgebra.mul!(x_temp::AbstractArray{T,2}, A::AbstractDerivativeOperator{T}, M::AbstractMatrix{T}) where T<:Real
    if size(x_temp) == reverse(size(M))
        for i = 1:size(M,1)
            mul!(view(x_temp,i,:), A, view(M,i,:))
        end
    else
        for i = 1:size(M,2)
            mul!(view(x_temp,:,i), A, view(M,:,i))
        end
    end
end

# Base.length(A::AbstractDerivativeOperator) = A.stencil_length
Base.ndims(A::AbstractDerivativeOperator) = 2
Base.size(A::AbstractDerivativeOperator) = (A.len, A.len + 2)
Base.size(A::AbstractDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::AbstractDerivativeOperator) = reduce(*, size(A))

#=
    For the evenly spaced grid we have a symmetric matrix
=#
Base.transpose(A::DerivativeOperator) = A
Base.adjoint(A::DerivativeOperator) = A
LinearAlgebra.issymmetric(::DerivativeOperator) = true

#=
    Fallback methods that use the full representation of the operator
=#
Base.exp(A::AbstractDerivativeOperator{T}) where T = exp(convert(A))
Base.:\(A::AbstractVecOrMat, B::AbstractDerivativeOperator) = A \ convert(Array,B)
Base.:\(A::AbstractDerivativeOperator, B::AbstractVecOrMat) = Array(A) \ B
Base.:/(A::AbstractVecOrMat, B::AbstractDerivativeOperator) = A / convert(Array,B)
Base.:/(A::AbstractDerivativeOperator, B::AbstractVecOrMat) = Array(A) / B

#=
    The Inf opnorm can be calculated easily using the stencil coeffiicents, while other opnorms
    default to compute from the full matrix form.
=#
function LinearAlgebra.opnorm(A::DerivativeOperator, p::Real=2)
    if p == Inf
        sum(abs.(A.stencil_coefs)) / A.dx^A.derivative_order
    else
        opnorm(BandedMatrix(A), p)
    end
end

########################################################################

get_type(::AbstractDerivativeOperator{T}) where {T} = T

function *(A::AbstractDerivativeOperator,x::AbstractVector)
    y = zeros(promote_type(eltype(A),eltype(x)), length(x)-2)
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, x::AbstractVector)
    return y
end


function *(A::AbstractDerivativeOperator,M::AbstractMatrix)
    y = zeros(promote_type(eltype(A),eltype(M)), size(A,1), size(M,2))
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, M::AbstractMatrix)
    return y
end


function *(M::AbstractMatrix,A::AbstractDerivativeOperator)
    y = zeros(promote_type(eltype(A),eltype(M)), size(M,1), size(A,2))
    LinearAlgebra.mul!(y, M, BandedMatrix(A))
    return y
end


function *(A::AbstractDerivativeOperator,B::AbstractDerivativeOperator)
    return BandedMatrix(A)*BandedMatrix(B)
end

################################################################################

function *(coeff_func::Function, A::DerivativeOperator{T,N,Wind}) where {T,N,Wind}
    coefficients = A.coefficients === nothing ? Vector{T}(undef,A.len) : A.coefficients
    DerivativeOperator{T,N,Wind,typeof(A.dx),typeof(A.stencil_coefs),
                       typeof(A.low_boundary_coefs),typeof(coefficients),
                       typeof(coeff_func)}(
        A.derivative_order, A.approximation_order,
        A.dx, A.len, A.stencil_length,
        A.stencil_coefs,
        A.boundary_stencil_length,
        A.boundary_point_count,
        A.low_boundary_coefs,
        A.high_boundary_coefs,coefficients,coeff_func
        )
end

################################################################################

function DiffEqBase.update_coefficients!(A::AbstractDerivativeOperator,u,p,t)
    if A.coeff_func !== nothing
        A.coeff_func(A.coefficients,u,p,t)
    end
end

################################################################################

(L::DerivativeOperator)(u,p,t) = L*u
(L::DerivativeOperator)(du,u,p,t) = mul!(du,L,u)
