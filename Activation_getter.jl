#Lots of the drivers and plotters have to extract the activations from the KAN-ODE
#for plotting, visualization, pruning, etc. This shared function enables this. 
function activation_getter(pM_new, kan1, grid_size)

    lay1=kan1[1]
    st=stM[1]
    pc1=pM_new.C
    pc1x1=pc1[:, 1:grid_size]
    pc1x2=pc1[:, grid_size+1:2*grid_size]
    pc1x3=pc1[:, 2*grid_size+1:3*grid_size]
    pc1x4=pc1[:, 3*grid_size+1:4*grid_size]

    pw1=pM_new.W
    pw1x1=pw1[:, 1]
    pw1x2=pw1[:, 2]
    pw1x3=pw1[:, 3]
    pw1x4=pw1[:, 4]

    size_in  = size(X)                          # [I, ..., batch,]

    x = reshape(X, lay1.in_dims, :)
    K = size(x, 2)

    x_norm = lay1.normalizer(x)              # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                        # [1, K]
    basis  = lay1.basis_func(x_resh, st.grid, lay1.denominator) # [G, I * K]
    basisx1=basis[:, 1:4:end] 
    basisx2=basis[:, 2:4:end] 
    basisx3=basis[:, 3:4:end] 
    basisx4=basis[:, 4:4:end] 
    activations_x1=basisx1'*pc1x1'
    activations_x2=basisx2'*pc1x2'
    activations_x3=basisx3'*pc1x3'
    activations_x4=basisx4'*pc1x4'
    activations_x1+=lay1.base_act.(x_norm[1, :]).*pw1x1'
    activations_x2+=lay1.base_act.(x_norm[2, :]).*pw1x2'
    activations_x3+=lay1.base_act.(x_norm[3, :]).*pw1x3'
    activations_x4+=lay1.base_act.(x_norm[4, :]).*pw1x4'

    ##sanity check: run the actual spline formulation and make sure they match
    #basis  = reshape(basis, lay1.grid_len * lay1.in_dims, K)    # [G * I, K]
    #spline = pc1*basis+pw1*lay1.base_act.(x)                                  # [O, K]
    #sum(abs.(spline.-((activations_x+activations_y)'[:, :])).<1e-10)==length(spline) #make sure it's all equal 
#=
    ##second layer
    LV_samples_lay1=kan1[1](X, pM_.layer_1, stM[1])[1] #this is the activation function results for the first layer

    x = reshape(LV_samples_lay1, lay2.in_dims, :)
    K = size(x, 2)

    x_norm = lay2.normalizer(x)              # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                        # [1, K]
    basis  = lay2.basis_func(x_resh, st.grid, lay2.denominator) # [G, I * K]
    activations_second=zeros(lay2.in_dims*2, K)
    for i in 1:lay2.in_dims
        basis_curr=basis[:, i:lay2.in_dims:end]
        pc_curr=pc2[:, (i-1)*grid_size+1:i*grid_size]
        activations_curr=basis_curr'*pc_curr'
        activations_curr+=(lay2.base_act.(x[i, :]).*pw2[:, i]')
        activations_second[2*i-1:2*i, :]=activations_curr'
    end
    ##sanity check: run the actual spline formulation and make sure they match 
    #basis  = reshape(basis, lay2.grid_len * lay2.in_dims, K)    # [G * I, K]
    #spline = pc2*basis+pw2*lay2.base_act.(x)                                  # [O, K]
    ##activation_compare=zeros(2, K)
    #activation_compare[1, :]=sum(activations_second[1:2:end, :], dims=1)
    #activation_compare[2, :]=sum(activations_second[2:2:end, :], dims=1)
    #sum(abs.(spline.-((activation_compare))).<1e-10)==length(spline) #make sure it's all equal 
    =#
    return activations_x1, activations_x2,activations_x3, activations_x4, K
end