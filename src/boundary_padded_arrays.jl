# Boundary Padded Arrays
abstract type AbstractBoundaryPaddedArray{T, N} <: AbstractArray{T, N} end
abstract type AbstractDirectionalBoundaryPaddedArray{T, N, D} <: AbstractBoundaryPaddedArray{T, N}
abstract type AbstractComposedBoundaryPaddedArray{T, N} <: AbstractBoundaryPaddedArray{T,N}
"""
A vector type that extends a vector u with ghost points at either end
"""
struct BoundaryPaddedVector{T,T2 <: AbstractVector{T}} <: AbstractBoundaryPaddedArray{T, 1}
    l::T
    r::T
    u::T2
end
Base.length(Q::BoundaryPaddedVector) = length(Q.u) + 2
Base.size(Q::BoundaryPaddedVector) = (length(Q.u) + 2,)
Base.lastindex(Q::BoundaryPaddedVector) = Base.length(Q)

function Base.getindex(Q::BoundaryPaddedVector,i)
    if i == 1
        return Q.l
    elseif i == length(Q)
        return Q.r
    else
        return Q.u[i-1]
    end
end

"""
Higher dimensional generalization of BoundaryPaddedVector, pads an array of dimension N along the dimension D with 2 Arrays of dimension N-1, stored in lower and upper

"""
struct BoundaryPaddedArray{T, D, N, M, V<:AbstractArray{T, N}, B<: AbstractArray{T, M}} <: AbstractDirectionalBoundaryPaddedArray{T,N, D}
    lower::B #an array of dimension M = N-1, used to extend the lower index boundary
    upper::B #Ditto for the upper index boundary
    u::V
end

getaxis(Q::BoundaryPaddedArray{T,D,N,M,V,B}) where {T,D,N,M,V,B} = D

function Base.size(Q::BoundaryPaddedArray)
    S = [size(Q.u)...]
    S[getaxis(Q)] += 2
    return Tuple(S)
end

"""
A = compose(padded_arrays::BoundaryPaddedArray...)

-------------------------------------------------------------------------------------

Example:
A = compose(Ax, Ay, Az) # 3D domain
A = compose(Ax, Ay) # 2D Domain

Composes BoundaryPaddedArrays that extend the same u for each different dimension that u has in to a ComposedBoundaryPaddedArray

Ax Ay and Az can be passed in any order, as long as there is exactly one BoundaryPaddedArray that extends each dimension.
"""
function compose(padded_arrays::BoundaryPaddedArray...)
    N = ndims(padded_arrays[1])
    Ds = getaxis.(padded_arrays)
    (length(padded_arrays) == N) || throw(ArgumentError("The padded_arrays must cover every dimension - make sure that the number of padded_arrays is equal to ndims(u)."))
    for D in Ds
        length(setdiff(Ds, D)) < N || throw(ArgumentError("There are multiple Arrays that extend along dimension $D - make sure every dimension has a unique extension"))
    end
    reduce((|), fill(padded_arrays[1].u, (length(padded_arrays),)) .== getfield.(padded_arrays, :u)) || throw(ArgumentError("The padded_arrays do not all extend the same u!"))
    padded_arrays = padded_arrays[sortperm([Ds...])]
    lower = [padded_array.lower for padded_array in padded_arrays]
    upper = [padded_array.upper for padded_array in padded_arrays]

    ComposedBoundaryPaddedArray{gettype(padded_arrays[1]),N,N-1,typeof(padded_arrays[1].u),typeof(lower[1])}(lower, upper, padded_arrays[1].u)
end

# Composed BoundaryPaddedArray

struct ComposedBoundaryPaddedArray{T, N, M, V<:AbstractArray{T, N}, B<: AbstractArray{T, M}} <: AbstractComposedBoundaryPaddedArray{T, N}
    lower::Vector{B}
    upper::Vector{B}
    u::V
end

# Aliases
AbstractBoundaryPaddedMatrix{T} = AbstractBoundaryPaddedArray{T,2}
AbstractBoundaryPadded3Tensor{T} = AbstractBoundaryPaddedArray{T,3}

BoundaryPaddedMatrix{T, D, V, B} = BoundaryPaddedArray{T, D, 2, 1, V, B}
BoundaryPadded3Tensor{T, D, V, B} = BoundaryPaddedArray{T, D, 3, 2, V, B}

ComposedBoundaryPaddedMatrix{T,V,B} = ComposedBoundaryPaddedArray{T,2,1,V,B}
ComposedBoundaryPadded3Tensor{T,V,B} = ComposedBoundaryPaddedArray{T,3,2,V,B}

Base.size(Q::ComposedBoundaryPaddedArray) = size(Q.u).+2

"""
Ax, Ay,... = decompose(A::ComposedBoundaryPaddedArray)

-------------------------------------------------------------------------------------

Decomposes a ComposedBoundaryPaddedArray in to components that extend along each dimension individually
"""
decompose(A::ComposedBoundaryPaddedArray) = Tuple([BoundaryPaddedArray{gettype(A), i, ndims(A), ndims(A)-1, typeof(lower[1])}(A.lower[i], A.upper[i], A.u) for i in 1:ndims(A)])


Base.length(Q::AbstractBoundaryPaddedArray) = reduce((*), size(Q))
Base.firstindex(Q::AbstractBoundaryPaddedArray, d::Int) = 1
Base.lastindex(Q::AbstractBoundaryPaddedArray) = length(Q)
Base.lastindex(Q::AbstractBoundaryPaddedArray, d::Int) = size(Q)[d]
gettype(Q::AbstractBoundaryPaddedArray{T,N}) where {T,N} = T
Base.ndims(Q::AbstractBoundaryPaddedArray{T,N}) where {T,N} = N

add_dims(A::AbstractArray, n::Int) = cat(ndims(a) + n, a)

function Base.getindex(Q::BoundaryPaddedArray{T,D,N,M,V,B}, _inds::Vararg{Int,N}) where {T,D,N,M,V,B} #supports range and colon indexing!
    inds = [_inds...]
    S = size(Q)
    dim = D
    otherinds = inds[setdiff(1:N, dim)]
    @assert length(S) == N
    if inds[dim] == 1
        return Q.lower[otherinds...]
    elseif inds[dim] == S[dim]
        return Q.upper[otherinds...]
    elseif typeof(inds[dim]) <: Integer
        inds[dim] = inds[dim] - 1
        return Q.u[inds...]
    elseif typeof(inds[dim]) == Colon
        if mapreduce(x -> typeof(x) != Colon, (|), otherinds)
            return vcat(Q.lower[otherinds...],  Q.u[inds...], Q.upper[otherinds...])
        else
            throw("A colon on the extended dim is as yet incompatible with additional colons")
        end
    elseif typeof(inds[dim]) <: AbstractArray
        throw("Range indexing not yet supported!")
    end
end

function Base.getindex(Q::ComposedBoundaryPaddedArray{T, N, M, V, B} , inds::Vararg{Int, N}) where {T, N, M, V, B} #as yet no support for range indexing or colon indexing
    S = size(Q)
    @assert reduce((&), inds .<= S)
    for (dim, index) in enumerate(inds)
        if index == 1
            _inds = inds[setdiff(1:N, dim)]
            if (1 ∈ _inds) | reduce((|), S[setdiff(1:N, dim)] .== _inds)
                return zero(T)
            else
                return Q.lower[dim][(_inds.-1)...]
            end
        elseif index == S[dim]
            _inds = inds[setdiff(1:N, dim)]
            if (1 ∈ _inds) | reduce((|), S[setdiff(1:N, dim)] .== _inds)
                return zero(T)
            else
                return Q.upper[dim][(_inds.-1)...]
            end
        end
     end
    return Q.u[(inds.-1)...]
end

# Multilayer

struct MultiLayerBoundaryPaddedArray{T, D, N, L, V<:AbstractArray{T, N}, B<: AbstractArray{T, N}} <: AbstractDirectionalBoundaryPaddedArray{T,N,D}
    lower::B #an array of dimension N, used to extend the lower index boundary
    upper::B #Ditto for the upper index boundary
    u::V
end

getaxis(Q::MultiLayerBoundaryPaddedArray{T,D,N,L,V,B}) where {T,D,N,L,V,B} = D
getlayers(Q::MultiLayerBoundaryPaddedArray{T,D,N,L,V,B}) where {T,D,N,L,V,B} = L

function Base.size(Q::MultiLayerBoundaryPaddedArray)
    S = [size(Q.u)...]
    S[getaxis(Q)] += 2*getlayers(Q)
    return Tuple(S)
end

function Base.getindex(Q::MultiLayerBoundaryPaddedArray{T,D,N,L,V,B}, _inds::Vararg{Int,N}) where {T,D,N,L,V,B} #supports range and colon indexing!

    S = size(Q)
    @assert length(S) == N
    if _inds[D] ≤ L
        I = CartesianIndex(_inds)
        return Q.lower[I]
    elseif _inds[D] ≥ S[D]-L+1
        return Q.upper[I]
    elseif typeof(_inds[D]) <: Integer
        inds[D] = inds[D] - 1
        return Q.u[I]
    elseif typeof(inds[D]) == Colon
        throw("A colon on the extended dim is as yet incompatible with additional colons")
    elseif typeof(inds[D]) <: AbstractArray
        throw("Range indexing not yet supported!")
    end
end

struct ComposedMultiLayerBoundaryPaddedArray{T, N, L, V<:AbstractArray{T, N}, B<: AbstractArray{T, N}} <: AbstractComposedBoundaryPaddedArray{T, N}
    lower::Vector{B}
    upper::Vector{B}
    u::V
end

function unit_indices(N::Int) #create unit CartesianIndex for each dimension
    out = Vector{CartesianIndex{N}}(undef, N)
    null = zeros(Int64, N)
    for i in 1:N
        unit_i = copy(null)
        unit_i[i] = 1
        out[i] = CartesianIndex(Tuple(unit_i))
    end
    Tuple(out)
end

or(B::BitArray)
    for b in B
        if b
            return true
        end
    end
    return false
end


function Base.getindex(Q::ComposedMultiLayerBoundaryPaddedArray{T, N, L, V, B} , inds::Vararg{Int, N}) where {T, N, L, V, B} #as yet no support for range indexing or colon indexing
    S = size(Q)
    @assert reduce((&), inds .<= S)
    I = CartesianIndex(inds)
    IL = CartesianIndex(L)
    ê = unit_indices(N)
    _ê = (fill(IL,N) .- L.*ê)
    l = [L...]
    for (dim, index) in enumerate(inds)
        _inds = inds[setdiff(1:N, dim)]
        _l = l[setdiff(1:N, dim)]
        is_corner = or([(_inds[i] ∈ 1:_l[i]) | (_inds[i] ∈ (perpsize(Q,dim)[i]-_l[i]+1):perpsize(Q,dim)[i]) for i in 1:N-1])
        if is_corner
            return zero(T)
        elseif index ≤ L
            return Q.lower[dim][I - _ê[dim]]
        elseif index ≥ S[dim]-L[dim]+1
            return Q.upper[dim][(I - _ê[dim] - (L[dim]+size(Q.u,dim))*ê[dim]]
        end
     end
    return Q.u[I-IL]
end

function compose(padded_arrays::Vararg{MultiLayerBoundaryPaddedArray{T,D,N,L,V,B}, N}) where {T,D,N,L,V,B}
    Ds = getaxis.(padded_arrays)
    for D in Ds
        length(setdiff(Ds, D)) < N || throw(ArgumentError("There are multiple Arrays that extend along dimension $D - make sure every dimension has a unique extension"))
    end
    reduce((|), fill(padded_arrays[1].u, (length(padded_arrays),)) .== getfield.(padded_arrays, :u)) || throw(ArgumentError("The padded_arrays do not all extend the same u!"))
    padded_arrays = padded_arrays[sortperm([Ds...])]
    lower = [padded_array.lower for padded_array in padded_arrays]
    upper = [padded_array.upper for padded_array in padded_arrays]
    L = getlayers.(padded_arrays)
    ComposedMultiLayerBoundaryPaddedArray{gettype(padded_arrays[1]), N, L, typeof(padded_arrays[1].u),typeof(lower[1])}(lower, upper, padded_arrays[1].u)
end

"""
A function to generate the correct permutation to flip an array of dimension `N` to be orthogonal to `dim`
"""
function orth_perm(N::Int, dim::Int)
    if dim == N
        return Vector(1:N)
    elseif dim < N
        P = experms(N, dim+1)
        P[dim], P[dim+1] = P[dim+1], P[dim]
        return P
    else
        throw("Dim is greater than N!")
    end
end

function Base.convert(MultiLayerBoundaryPaddedArray, A::BoundaryPaddedArray{T,D,N,M,V,B}) where {T,D,N,L,V,B}
    lower = permutedims(add_dims(A.lower, 1), orth_perm(N, D))
    upper = permutedims(add_dims(A.upper, 1), orth_perm(N, D))
    MultiLayerBoundaryPaddedArray{gettype(A),getaxis(A),ndims(A),1,typeof(A.u), typeof(lower)}(lower,upper,A.u)
end
Base.convert(T, A::MultiLayerBoundaryPaddedArray) = A

function compose(padded_arrays::Vararg{AbstractDirectionalBoundaryPaddedArray, N}) where N
    compose(convert.(fill(MultiLayerBoundaryPaddedArray,N), padded_arrays)...)
end

decompose(A::ComposedMultiLayerBoundaryPaddedArray) = Tuple([MultiLayerBoundaryPaddedArray{gettype(A), i, ndims(A), getlayers(A)[i], typeof(lower[1])}(A.lower[i], A.upper[i], A.u) for i in 1:ndims(A)])
