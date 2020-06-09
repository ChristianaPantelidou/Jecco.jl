
abstract type Filter{T,N} end

function KO_kernel(order::Int, sigma_diss::T) where {T<:Real}
    if order == 3
        kernel = -T[1, -4, 6, -4, 1]
        fac = sigma_diss / 16
    elseif order == 5
        kernel = T[1, -6, 15, -20, 15, -6, 1]
        fac = sigma_diss / 64
    else
        error("order = $order Kreiss-Oliger filter not yet implemented.")
    end

    rmul!(kernel, fac)

    # adding one to the middle entry so that it becomes a low-pass filter
    mid = div(length(kernel) + 1, 2)
    kernel[mid] += 1

    kernel
end

function exp_kernel!(f::Vector{T}, α::T, γ::T) where {T<:Real}
    M  = length(f) - 1
    f .= [ exp( -α * ( (i-1)/M )^(γ*M) ) for i in eachindex(f) ]
    nothing
end
function exp_kernel(dim::Int, α::T, γ::T) where {T<:Real}
    f  = Array{T}(undef, dim)
    exp_kernel!(f, α, γ)
    f
end


struct ConvFilter{T<:Real,N,A} <: Filter{T,N}
    kernel :: Vector{T}
    _cache :: A
end

struct FftFilter{T<:Real,N,A,FT<:FFTW.r2rFFTWPlan} <: Filter{T,N}
    kernel   :: Vector{T}
    fft_plan :: FT
    _cache   :: A
end

struct KO_Filter{N} end

function KO_Filter{N}(order::Int, sigma_diss::T, Nxx...) where {T<:Real,N}
    kernel = KO_kernel(order, sigma_diss)
    _cache = Array{T}(undef, Nxx...)
    ConvFilter{T,N,typeof(_cache)}(kernel, _cache)
end

KO_Filter(args...) = KO_Filter{1}(args...)


struct Exp_Filter{N} end

function Exp_Filter{N}(α::T, γ::T, Nxx...) where {T<:Real,N}
    kernel = exp_kernel(Nxx[N], α, γ)
    _cache = Array{T}(undef, Nxx...)

    # REDFT00 is the DCT-I, see http://www.fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028cosine_002fsine-transforms_0029.html#Real-even_002fodd-DFTs-_0028cosine_002fsine-transforms_0029

    # we want to use the DCT-I since this basis is precisely the one we have by
    # using the Gauss-Lobatto grid points
    fft_plan = FFTW.plan_r2r(_cache, FFTW.REDFT00, N)

    FftFilter{T,N,typeof(_cache),typeof(fft_plan)}(kernel, fft_plan, _cache)
end

FftFilter(args...) = FftFilter{1}(args...)


function convolution!(fout::AbstractVector{T}, f::AbstractVector{T},
                      g::AbstractVector{T}) where {T}
    f_len = length(f)
    g_len = length(g)
    mid   = div(g_len + 1, 2)

    @fastmath @inbounds for i in eachindex(f)
        sum_i = zero(T)

        if mid <= i <= (f_len - mid + 1)
            @inbounds for aa in 1:g_len
                i_circ = i - (mid - aa)
                sum_i += g[aa] * f[i_circ]
            end
        else
            @inbounds for aa in 1:g_len
                # imposing periodicity
                i_circ = 1 + mod(i - (mid-aa) - 1, f_len)
                sum_i += g[aa] * f[i_circ]
            end
        end
        fout[i] = sum_i
    end

    fout
end

# convolution along N axis
function convolution!(fout::AbstractArray{T,M}, f::AbstractArray{T,M},
                      g::AbstractVector{T}, N::Integer) where {T,M}
    # make sure axis of convolution is contained in the dimensions of f
    @assert N <= M

    f_len = length(axes(f,N))
    g_len = length(g)
    mid   = div(g_len + 1, 2)

    @fastmath @inbounds for idx in CartesianIndices(f)
        i     = idx.I[N] # convolution to be done along this direction
        sum_i = zero(T)

        if mid <= i <= (f_len - mid + 1)
            @inbounds for aa in 1:g_len
                i_circ = i - (mid - aa)
                I = Base.setindex(idx.I, i_circ, N)
                sum_i += g[aa] * f[I...]
            end
        else
            @inbounds for aa in 1:g_len
                # imposing periodicity
                i_circ = 1 + mod(i - (mid-aa) - 1, f_len)
                I = Base.setindex(idx.I, i_circ, N)
                sum_i += g[aa] * f[I...]
            end
        end
        fout[idx] = sum_i
    end

    fout
end

function (filter::ConvFilter)(f::AbstractVector)
    @assert length(filter._cache) == length(f)
    convolution!(filter._cache, f, filter.kernel)
    copyto!(f, filter._cache)
end

function (filter::ConvFilter{T,M})(f::AbstractArray{T}) where {T,M}
    @assert length(filter._cache) == length(f)
    convolution!(filter._cache, f, filter.kernel, M)
    copyto!(f, filter._cache)
end


# see, eg, "Idempotent filtering in spectral and spectral element methods",
# Journal of Computational Physics 220 (2006) 41-58
function (filter::FftFilter)(f::AbstractVector)
    M = length(f) - 1

    copyto!(filter._cache, f)

    # compute the DCT-I of f
    mul!(f, filter.fft_plan, filter._cache)

    # in momentum space, act with the filter kernel [eq (2.5) of the paper above]
    # 0.5/M .* filter.kernel .* f
    @inbounds @simd for i in eachindex(f)
        filter._cache[i] = 0.5/M * filter.kernel[i] * f[i]
    end

    # now go back to position space using the DCT-I. the division by 2*(N-1)
    # above comes from the normalization used. DCT-I is its own inverse up to a
    # constant, and FFTW defines it as such, cf:
    # http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029
    mul!(f, filter.fft_plan, filter._cache)
end
