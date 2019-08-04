
abstract type MultiDimensionalBC{T, N} <: AbstractBC{T} end


@noinline function _slice_rmul!(u_temp::AbstractArray{T,N}, A::AbstractDiffEqLinearOperator, u::AbstractArray{T,N}, dim::Int, pre, post) where {T,N}
    for J in post
        for I in pre
            u_temp[I, :, J] = A*u[I, :, J]
        end
    end
    u_temp
end

function slice_rmul(A::AbstractDiffEqLinearOperator, u::AbstractArray{T,N}, dim::Int) where {T,N}
    @assert N != 1
    u_temp = similar(u)

    _slice_rmul!(u_temp, A, u, dim, CartesianIndices(axes(u)[1:dim-1]), CartesianIndices(axes(u)[(dim+1):end]))

    return u_temp
end

@noinline function _slice_rmul!(lower::AbstractArray, upper::AbstractArray, A::AbstractArray{B,M}, u::AbstractArray{T,N}, dim::Int, pre, post) where {T,B,N,M}
    for J in post
        for I in pre

            tmp = A[I,J]*u[I, :, J]
            lower[I,J], upper[I,J] = tmp.l, tmp.r
        end
    end
    return (lower, upper)
end

function slice_rmul(A::AbstractArray{B,M}, u::AbstractArray{T,N}, dim::Int) where {T, B, N,M}
    @assert N != 1
    @assert M == N-1
    lower = zeros(T,perpsize(u,dim))
    upper = zeros(T,perpsize(u,dim))

    _slice_rmul!(lower, upper, A, u, dim, CartesianIndices(axes(u)[1:dim-1]), CartesianIndices(axes(u)[(dim+1):end]))

    return (lower, upper)
end

"""
slicemul is the only limitation on the BCs here being used up to arbitrary dimension, an N dimensional implementation is needed.
"""

struct MultiDimDirectionalBC{T<:Number, B<:AtomicBC{T}, D, N, M} <: MultiDimensionalBC{T, N}
    BCs::Array{B,M} #dimension M=N-1 array of BCs to extend dimension D
end

struct ComposedMultiDimBC{T, B<:AtomicBC{T}, N,M} <: MultiDimensionalBC{T, N}
    BCs::Vector{Array{B, M}}
end

"""
A multiple dimensional BC, supporting arbitrary BCs at each boundary point.
To construct an arbitrary BC, pass an Array of BCs with dimension `N-1`, if `N` is the dimensionality of your domain `u`
with a size of `size(u)[setdiff(1:N, dim)]`, where dim is the dimension orthogonal to the boundary that you want to extend.

It is also possible to call
    Q_dim = MultiDimBC(YourBC, size(u), dim)
to use YourBC for the whole boundary orthogonal to that dimension.

Further, it is possible to call
Qx, Qy, Qz... = MultiDimBC(YourBC, size(u))
to use YourBC for the whole boundary for all dimensions. Valid for any number of dimensions greater than 1.
However this is only valid for Robin/General type BCs (including neummann/dirichlet) when the grid steps are equal in each dimension - including uniform grid case.

In the case where you want to extend the same Robin/GeneralBC to the whole boundary with a non unifrom grid, please use
    Qx, Qy, Qz... = RobinBC(l, r, (dx::Vector, dy::Vector, dz::Vector ...), approximation_order, size(u))
or
    Qx, Qy, Qz... = GeneralBC(αl, αr, (dx::Vector, dy::Vector, dz::Vector ...), approximation_order, size(u))

There are also constructors for NeumannBC, DirichletBC and Dirichlet0BC. Simply replace `dx` in the call with the tuple dxyz... as above, and append `size(u)`` to the argument signature.
The order is a required argument in this case.

where dx, dy, and dz are vectors of grid steps.

For Neumann0BC, please use
    Qx, Qy, Qz... = Neumann0BC(T::Type, (dx::Vector, dy::Vector, dz::Vector ...), approximation_order, size(u))
where T is the element type of the domain to be extended
"""
MultiDimBC(BC::Array{B,N}, dim::Integer) where {N, B<:AtomicBC} = MultiDimDirectionalBC{gettype(BC[1]), B, dim, N+1, N}(BC)
#s should be size of the domain
MultiDimBC(BC::B, s, dim::Integer) where  {B<:AtomicBC} = MultiDimDirectionalBC{gettype(BC), B, dim, length(s), length(s)-1}(fill(BC, s[setdiff(1:length(s), dim)]))

#Extra constructor to make a set of BC operators that extend an atomic BC Operator to the whole domain
#Only valid in the uniform grid case!
MultiDimBC(BC::B, s) where {B<:AtomicBC} = Tuple([MultiDimDirectionalBC{gettype(BC), B, dim, length(s), length(s)-1}(fill(BC, s[setdiff(1:length(s), dim)])) for dim in 1:length(s)])

# Additional constructors for cases when the BC is the same for all boundarties

PeriodicBC{T}(s) where T = MultiDimBC(PeriodicBC{T}(), s)

NeumannBC(α::AbstractVector{T}, dxyz, order, s) where T = RobinBC([zero(T), one(T), α[1]], [zero(T), one(T), α[2]], dxyz, order, s)
DirichletBC(αl::T, αr::T, s) where T = RobinBC([one(T), zero(T), αl], [one(T), zero(T), αr], [ones(T, si) for si in s], 2.0, s)

Dirichlet0BC(T::Type, s) = DirichletBC(zero(T), zero(T), s)
Neumann0BC(T::Type, dxyz, order, s) = NeumannBC([zero(T), zero(T)], dxyz, order, s)

RobinBC(l::AbstractVector{T}, r::AbstractVector{T}, dxyz, order, s) where {T} = Tuple([MultiDimDirectionalBC{T, RobinBC{T}, dim, length(s), length(s)-1}(fill(RobinBC(l, r, dxyz[dim], order), s[setdiff(1:length(s), dim)])) for dim in 1:length(s)])
GeneralBC(αl::AbstractVector{T}, αr::AbstractVector{T}, dxyz, order, s) where {T} = Tuple([MultiDimDirectionalBC{T, GeneralBC{T}, dim, length(s), length(s)-1}(fill(GeneralBC(αl, αr, dxyz[dim], order), s[setdiff(1:length(s), dim)])) for dim in 1:length(s)])


perpsize(A::AbstractArray{T,N}, dim::Integer) where {T,N} = size(A)[setdiff(1:N, dim)] #the size of A perpendicular to dim

"""
Q = compose(BCs...)

-------------------------------------------------------------------------------------

Example:
Q = compose(Qx, Qy, Qz) # 3D domain
Q = compose(Qx, Qy) # 2D Domain

Creates a ComposedMultiDimBC operator, Q, that extends every boundary when applied to a `u` with compatible size and number of dimensions.

Qx Qy and Qz can be passed in any order, as long as there is exactly one BC operator that extends each dimension.
"""
function compose(BCs...)
    T = gettype(BCs[1])
    N = ndims(BCs[1])
    Ds = getaxis.(BCs)
    (length(BCs) == N) || throw(ArgumentError("There must be enough BCs to cover every dimension - check that the number of MultiDimBCs == N"))
    for D in Ds
        length(setdiff(Ds, D)) == (N-1) || throw(ArgumentError("There are multiple boundary conditions that extend along $D - make sure every dimension has a unique extension"))
    end
    BCs = BCs[sortperm([Ds...])]

    ComposedMultiDimBC{T, Union{eltype.(BC.BCs for BC in BCs)...}, N,N-1}([condition.BCs for condition in BCs])
end

"""
Qx, Qy,... = decompose(Q::ComposedMultiDimBC{T,N,M})

-------------------------------------------------------------------------------------

Decomposes a ComposedMultiDimBC in to components that extend along each dimension individually
"""
decompose(Q::ComposedMultiDimBC) = Tuple([MultiDimBC(Q.BC[i], i) for i in 1:ndims(Q)])

getaxis(Q::MultiDimDirectionalBC{T, B, D, N, K}) where {T, B, D, N, K} = D
getboundarytype(Q::MultiDimDirectionalBC{T, B, D, N, K}) where {T, B, D, N, K} = B

Base.ndims(Q::MultiDimensionalBC{T,N}) where {T,N} = N

function Base.:*(Q::MultiDimDirectionalBC{T, B, D, N, K}, u::AbstractArray{T, N}) where {T, B, D, N, K}
    @assert perpsize(u, D) == size(Q.BCs) "Size of the BCs array in the MultiDimBC is incorrect, needs to be $(perpsize(u,D)) to extend dimension $D, got $(size(Q.BCs))"
    lower, upper = slice_rmul(Q.BCs, u, D)
    return BoundaryPaddedArray{T, D, N, K, typeof(u), typeof(lower)}(lower, upper, u)
end

function Base.:*(Q::ComposedMultiDimBC{T, B, N, K}, u::AbstractArray{T, N}) where {T, B, N, K}
    for dim in 1:N
        @assert perpsize(u, dim) == size(Q.BCs[dim]) "Size of the BCs array for dimension $dim in the MultiDimBC is incorrect, needs to be $(perpsize(u,dim)), got $(size(Q.BCs[dim]))"
    end
    out = slice_rmul.(Q.BCs, fill(u, N), 1:N)
    return ComposedBoundaryPaddedArray{T, N, K, typeof(u), typeof(out[1][1])}([A[1] for A in out], [A[2] for A in out], u)
end
