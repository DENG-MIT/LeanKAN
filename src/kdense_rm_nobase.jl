#
#======================================================#
# Kolmogorov-Arnold Layer
#======================================================#
@concrete struct KDense_rm_nobase{use_base_act} <: LuxCore.AbstractLuxLayer 
    in_dims::Int
    out_dims::Int
    grid_len::Int
    #
    normalizer
    mult_flag
    grid_lims
    denominator
    #
    basis_func
    base_act
    init_C
    init_W
end

function KDense_rm_nobase(
    in_dims::Int,
    out_dims::Int,
    grid_len::Int;
    #
    normalizer = tanh,
    mult_flag=2,
    grid_lims::NTuple{2, Real} = (-1.0f0, 1.0f0),
    denominator = Float32(2 / (grid_len - 1)),
    #
    basis_func = rbf,
    #
    base_act = swish,
    use_base_act = true,
    #
    init_C = glorot_uniform,
    init_W = glorot_uniform,
    allow_fast_activation::Bool = true,
)
    T = promote_type(eltype.(grid_lims)...)

    if isnothing(grid_lims)
        grid_lims = if normalizer ∈ (sigmoid, sigmoid_fast)
            (0, 1)
        elseif normalizer ∈ (tanh, tanh_fast, softsign)
            (-1, 1)
        else
            (-1, 1)
        end
    end

    grid_span =  grid_lims[2] > grid_lims[1]
    @assert grid_span > 0

    if isnothing(denominator)
        denominator = grid_span / (grid_len - 1)
    end

    if allow_fast_activation
        basis_func = NNlib.fast_act(basis_func)
        base_act = NNlib.fast_act(base_act)
        if normalizer!=false
            normalizer = NNlib.fast_act(normalizer)
        end
    end

    KDense_rm_nobase{use_base_act}(
        in_dims, out_dims, grid_len,
        normalizer, mult_flag, T.(grid_lims), T(denominator),
        basis_func, base_act, init_C, init_W,
    )
end

function LuxCore.initialparameters(
    rng::AbstractRNG,
    l::KDense_rm_nobase{use_base_act}
) where{use_base_act}
    p = (;
        C = l.init_C(rng, l.out_dims, l.grid_len * l.in_dims), # [O, G, I]
    )

    if use_base_act
        p = (;
            p...,
            W = l.init_W(rng, l.out_dims, l.in_dims),
        )
    end

    p
end

function LuxCore.initialstates(::AbstractRNG, l::KDense_rm_nobase,)
    (;
        grid = collect(LinRange(l.grid_lims..., l.grid_len))
    )
end

function LuxCore.statelength(l::KDense_rm_nobase)
    l.grid_len
end

function LuxCore.parameterlength(
    l::KDense_rm_nobase{use_base_act},
) where{use_base_act}
    len = l.in_dims * l.grid_len * l.out_dims
    if use_base_act
        len += l.in_dims * l.out_dims
    end

    len
end

function (l::KDense_rm_nobase{use_base_act})(x::AbstractArray, p, st) where{use_base_act}
    size_in  = size(x)                          # [I, ..., batch,]
    size_out = (l.out_dims, size_in[2:end]...,) # [O, ..., batch,]

    x       = reshape(x, l.in_dims, :)
    K       = size(x, 2)
    x_norm  = _broadcast(l.normalizer, x)                  # ∈ [-1, 1]
    x_resh  = reshape(x_norm, 1, :)                        # [1, K]

    basis   = l.basis_func(x_resh, st.grid, l.denominator) # [G, I * K]
    basis   = reshape(basis, l.grid_len * l.in_dims, K)    # [G * I, K]

    mult_index          = l.mult_flag
    spline_add_portion  = p.C[:, mult_index*l.grid_len+1:end]*basis[mult_index*l.grid_len+1:end]
    weighted_basis      = p.C[:, 1:mult_index*l.grid_len].*basis[1:mult_index*l.grid_len]'
    y                   = spline_add_portion + prod(reshape(sum(reshape(weighted_basis', (l.grid_len, mult_index*l.out_dims)), dims = 1), (:,l.out_dims)), dims=1)[:]

    reshape(y, size_out), st
end

