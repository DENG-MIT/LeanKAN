#
#======================================================#
# Kolmogorov-Arnold Layer
#======================================================#
@concrete struct KDense{use_base_act} <: LuxCore.AbstractLuxLayer 
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

function KDense(
    in_dims::Int,
    out_dims::Int,
    grid_len::Int;
    #
    normalizer = tanh,
    mult_flag=0,
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

    KDense{use_base_act}(
        in_dims, out_dims, grid_len,
        normalizer, mult_flag, T.(grid_lims), T(denominator),
        basis_func, base_act, init_C, init_W,
    )
end

function LuxCore.initialparameters(
    rng::AbstractRNG,
    l::KDense{use_base_act}
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

function LuxCore.initialstates(::AbstractRNG, l::KDense,)
    (;
        grid = collect(LinRange(l.grid_lims..., l.grid_len))
    )
end

function LuxCore.statelength(l::KDense)
    l.grid_len
end

function LuxCore.parameterlength(
    l::KDense{use_base_act},
) where{use_base_act}
    len = l.in_dims * l.grid_len * l.out_dims
    if use_base_act
        len += l.in_dims * l.out_dims
    end

    len
end

function (l::KDense{use_base_act})(x::AbstractArray, p, st) where{use_base_act}
    size_in  = size(x)                          # [I, ..., batch,]
    size_out = (l.out_dims, size_in[2:end]...,) # [O, ..., batch,]

    x = reshape(x, l.in_dims, :)
    K = size(x, 2)

    if l.normalizer!=false
        x_norm = _broadcast(l.normalizer, x)                  # ∈ [-1, 1]
    else
        x_norm=x
    end                  # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                        # [1, K]
    basis  = l.basis_func(x_resh, st.grid, l.denominator) # [G, I * K]
    basis  = reshape(basis, l.grid_len * l.in_dims, K)    # [G * I, K]
    #spline = p.C * basis                                  # [O, K]
    #for add kan with 7 inputs and 6 nodes of 5 grid each:
        #5 grid points are on our grid.
        #For a given input, we have to evaluate the 5 grid points...
        #once for each of the 6 nodes. So 30 total weights for a single input
        #times 7 inputs equals 210 weights. So p.C is 210
        #in the shape of 35x6, because 7 inputs on 5 grids, for each of 7 nodes.

    #For multKAN, just do yes/no for simplicity here:
    if l.mult_flag!=0
            ####for mult_add code, let's do mult with half rounded up, add with half rounded down
        mult_index=l.mult_flag
            ####now do addition spline on just the remaining indices,
            ####i.e. for a 7-input, 5-grid, 5-output, mult_index is 4 (half of input rounded up),
            #### so this takes just the last 7-4=3 input portions and runs the adding.
            ####where for a 5-grid, the "last 3" is the last 15 (bc 5*3).
            ####So spline_add_portion is STILL length 5 for 5 outputs,
            ####but only comes from summing the last 3 input activations, rather than all 7.
        if use_base_act
            base = p.W .* l.base_act.(x)'
        else
            base=zeros(l.out_dims, l.in_dims)
        end
        spline_add_portion=p.C[:, mult_index*l.grid_len+1:end]*basis[mult_index*l.grid_len+1:end]+sum(base[:, mult_index+1:end], dims=2)
            ####and then run the standard mult code on the remaining indices.
        weighted_basis=p.C[:, 1:mult_index*l.grid_len].*basis[1:mult_index*l.grid_len]' #pc1 weights applied to the basis functions. No addition.
        #weighted basis is (# outputs)x(# inputs * gridsize)

        #We want the (# inputs)*(# outputs) actual splines now.
        #so we sum over the grid now.
        #where basis is grid*input, and [:, 1] is the 5 grids for input 1
        #and so reshaped basis is just all input 1, then all input 2, stacked on top, etc.
        #so finally p.C each row is all input 1 on left, then all input 2, etc.
        #so to sum each input over its grid...
        summed_basis=sum(weighted_basis[:, 1:l.grid_len], dims=2)+base[:, 1]
        for i in 2:(mult_index) ####mult_index now, rather than l.in_dims, bc we cut off the last few for addition
            curr_low=(i-1)*l.grid_len+1
            curr_high=curr_low+l.grid_len-1
            summed_basis=summed_basis.* (sum(weighted_basis[:, curr_low:curr_high], dims=2)+base[:,i])
            #summed_basis=hcat(summed_basis, sum(weighted_basis[:, curr_low:curr_high], dims=2))
        end
        #spline=Matrix{typeof(p.C)}(undef, l.out_dims)
        #spline=prod(summed_basis[:, 1])
        #for i in 2:l.out_dims #### this remains the same bc same # of outputs regardless of mult/add breakup
        #    spline=vcat(spline, prod(summed_basis[i, :]))
        #end
        spline=summed_basis+spline_add_portion
    else
        spline = p.C * basis  
        if use_base_act
            base = p.W * l.base_act.(x)
            spline=spline+base
        end
    end

    y=spline
    reshape(y, size_out), st
end

