using Random, Lux, LinearAlgebra
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using BenchmarkTools
using OrdinaryDiffEq, Plots, DiffEqFlux, ForwardDiff
using Flux: Adam, mae, update!
using Flux
using MAT
using Plots
using ProgressBars
using Zygote: gradient as Zgrad
using LaTeXStrings


#this is a fix for an issue with an author's computer. Feel free to remove.
ENV["GKSwstype"] = "100"

# Directories
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
add_path    = "results/"
figpath=dir*add_path*"figs"
ckptpath=dir*add_path*"checkpoints"
mkpath(figpath)
mkpath(ckptpath)

#Load the correct KAN package
include("src_v2/KolmogorovArnold_mult_add.jl")
using .KolmogorovArnold_mult_add
include("Activation_getter.jl")

n_plot_save=1000
rng = Random.default_rng()
Random.seed!(rng, 0)

#inputs: A, B, C, D. All on [0, 1]
A=shuffle(collect(0:0.005:1))
B=shuffle(collect(0:0.005:1))
C=shuffle(collect(0:0.005:1))
D=shuffle(collect(0:0.005:1))
X=hcat(A, B, C, D)
prob_size=size(X)[1]
Y=zeros(size(X))
for i in 1:prob_size
    Y[i, 1]=X[i, 1]*X[i,2]
    Y[i, 2]=X[i, 3]*X[i,4]
    Y[i, 3]=X[i, 1]*X[i,2]
    Y[i, 4]=X[i, 3]*X[i,4]
end
X_train=X[1:150, :] #150 train, 50 test
X_test=X[151:200, :]
Y_train=Y[1:150, :]
Y_test=Y[151:200, :]
train_size=150
test_size=50

basis_func = rbf      # rbf, rswaf
normalizer = tanh_fast # sigmoid(_fast), tanh(_fast), softsign
kan1 = Lux.Chain(
    KDense(4,  4, 4; use_base_act = true, basis_func, normalizer, mult_flag=2),
    #In order: 4 inputs, 4 outputs, 4 grid points. mult_flag encodes n^mu=2 (i.e. half and half multiplication and addition split)
)
pM , stM  = Lux.setup(rng, kan1)

pM_data = getdata(ComponentArray(pM))
pM_axis = getaxes(ComponentArray(pM))
p = (deepcopy(pM_data))./1e3


#solver functions

function predict(p::AbstractVector{T}) where T
    Y_kan=zeros(T,size(X_train)) #because Y_kan needs to be able to handle duals, so make zeros of the same type T as p (so it can handle both float32 AND dual)
    for i in 1:train_size
        Y_kan[i, :]=kan1(X_train[i, :], ComponentArray(p, pM_axis), stM)[1]
    end
    return Y_kan
end

function predict_test(p)
    Y_kan=zeros(size(X_test))
    for i in 1:test_size
        Y_kan[i, :]=kan1(X_test[i, :], ComponentArray(p, pM_axis), stM)[1]
    end
    return Y_kan
end


function loss(p)
    Y_kan=predict(p)
    return sum(abs.(Y_kan.-Y_train).^2)
end

function loss_report(p)
    Y_kan=predict(p)
    loss=Y_kan.-Y_train
    loss_1=sum(abs.(loss[:, 1]).^2)/train_size
    loss_2=sum(abs.(loss[:, 2]).^2)/train_size
    loss_3=sum(abs.(loss[:, 3]).^2)/train_size
    loss_4=sum(abs.(loss[:, 4]).^2)/train_size
    return loss_1, loss_2, loss_3, loss_4
end
function loss_report_test(p)
    Y_kan=predict_test(p)
    loss=Y_kan.-Y_test
    loss_1=sum(abs.(loss[:, 1]).^2)/test_size
    loss_2=sum(abs.(loss[:, 2]).^2)/test_size
    loss_3=sum(abs.(loss[:, 3]).^2)/test_size
    loss_4=sum(abs.(loss[:, 4]).^2)/test_size
    return loss_1, loss_2, loss_3, loss_4
end

line_style=:solid
#=
function plot_save(train_losses_1, train_losses_2, train_losses_3, train_losses_4, test_losses_1, test_losses_2, test_losses_3, test_losses_4, epoch)

    plt=Plots.plot(train_losses_1, yaxis=:log, c=:forestgreen, label="AB outputs (1 and 3)", dpi=600, size=(300, 210), linestyle=line_style)
    plot!(train_losses_2, yaxis=:log, c=:grey0, label="CD outputs (2 and 4)")
    plot!(train_losses_3, yaxis=:log, c=:forestgreen, label=false, linestyle=line_style)
    plot!(train_losses_4, yaxis=:log, c=:grey0, label=false)
    plot!(test_losses_1, yaxis=:log, c=:forestgreen, label=false, linestyle=line_style)
    plot!(test_losses_2, yaxis=:log, c=:grey0, label=false)
    plot!(test_losses_3, yaxis=:log, c=:forestgreen, label=false, linestyle=line_style)
    plot!(test_losses_4, yaxis=:log, c=:grey0, label=false)
    xlabel!("Epoch")
    ylabel!("Loss")
    png(plt, string(figpath, "/loss_v2.png"))
end
=#
function plot_save(train_losses_1, train_losses_2, train_losses_3, train_losses_4, test_losses_1, test_losses_2, test_losses_3, test_losses_4, epoch)

    plt=Plots.plot(train_losses_1, yaxis=:log, c=:forestgreen, label="\$z_1\$ train", legend=false, dpi=600, size=(250, 140), linestyle=line_style)
    plot!(train_losses_2, yaxis=:log, c=:grey30, label="\$z_2\$ train")
    plot!(train_losses_3, yaxis=:log, c=:mediumseagreen, label="\$z_3\$ train", linestyle=line_style)
    plot!(train_losses_4, yaxis=:log, c=:snow4, label="\$z_4\$ train", ylims=[1e-5, 1], framestyle = :box, grid=false)
    xlabel!("Epoch")
    ylabel!("Loss")
    png(plt, string(figpath, "/loss_v2.png"))
end
# TRAINING
opt = Flux.Adam(1e-3)

N_iter = 3001
i_current = 1
train_losses_1=[]
train_losses_2=[]
train_losses_3=[]
train_losses_4=[]
test_losses_1=[]
test_losses_2=[]
test_losses_3=[]
test_losses_4=[]


##Actual training loop:
iters=tqdm(1:N_iter-i_current)
 for i in iters
    global i_current
    
    # gradient computation
    grad = ForwardDiff.gradient(loss, p)

    #model update
    Flux.update!(opt, p, grad)

    #loss metrics
    loss_curr=loss(p)
    losses_curr=deepcopy(loss_report(p))
    losses_curr_test=deepcopy(loss_report_test(p)) 
    append!(train_losses_1, [losses_curr[1]])
    append!(train_losses_2, [losses_curr[2]])
    append!(train_losses_3, [losses_curr[3]])
    append!(train_losses_4, [losses_curr[4]])
    append!(test_losses_1, [losses_curr_test[1]])
    append!(test_losses_2, [losses_curr_test[2]])
    append!(test_losses_3, [losses_curr_test[3]])
    append!(test_losses_4, [losses_curr_test[4]])

    set_description(iters, string("Loss:", loss_curr))
    i_current = i_current + 1


    if i%n_plot_save==0
        plot_save(train_losses_1, train_losses_2, train_losses_3, train_losses_4, test_losses_1, test_losses_2, test_losses_3, test_losses_4, i)
    end

    
end




beta=100000
top_marg=-1.91Plots.mm
bot_marg=-6.4Plots.mm
left_marg=-6.638Plots.mm
right_marg=-1.88Plots.mm
sf=2
scalefontsizes()
scalefontsizes(1/sf)
width=70
height=75

x1=collect(0:0.01:1)
x2=collect(0:0.01:1)
x3=collect(0:0.01:1)
x4=collect(0:0.01:1)
acts_x1=zeros(length(x1), 4)
acts_x2=zeros(length(x2), 4)
acts_x3=zeros(length(x3), 4)
acts_x4=zeros(length(x4), 4)

for i in 1:length(x1)
    global X=[x1[i], x2[i], x3[i], x4[i]]
    acts_x1[i, :], acts_x2[i, :], acts_x3[i, :], acts_x4[i, :] = activation_getter(ComponentArray(p, pM_axis).layer_1, kan1, 4)
end

for i in 1:4 #for each of 4 nodes, there is an X1, X2, X3, X4 plot..
    input_range=1 #based on problem statement
    output_range=maximum(acts_x1[:, i])-minimum(acts_x1[:, i])
    acts_scale=output_range/input_range
    trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
    plt=Plots.plot(x1, acts_x1[:, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(x1), sigdigits=1), floor(maximum(x1), sigdigits=1)], yticks=[-1, 0], ylims=[-1, 0.5], alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
    png(plt, string(dir, "activation_plots/X1", string(i), ".png"))

    output_range=maximum(acts_x2[:, i])-minimum(acts_x2[:, i])
    acts_scale=output_range/input_range
    trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
    plt=Plots.plot(x1, acts_x2[:, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(x1), sigdigits=1), floor(maximum(x1), sigdigits=1)], yticks=[-1, 0], ylims=[-1, 0.5], alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
    png(plt, string(dir, "activation_plots/X2", string(i), ".png"))

    output_range=maximum(acts_x3[:, i])-minimum(acts_x3[:, i])
    acts_scale=output_range/input_range
    trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
    plt=Plots.plot(x1, acts_x3[:, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(x1), sigdigits=1), floor(maximum(x1), sigdigits=1)], yticks=[-1, 0], ylims=[-1, 0.5], alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
    png(plt, string(dir, "activation_plots/X3", string(i), ".png"))

    output_range=maximum(acts_x4[:, i])-minimum(acts_x4[:, i])
    acts_scale=output_range/input_range
    trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
    plt=Plots.plot(x1, acts_x4[:, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(x1), sigdigits=1), floor(maximum(x1), sigdigits=1)], yticks=[-1, 0], ylims=[-1, 0.5], alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
    png(plt, string(dir, "activation_plots/X4", string(i), ".png"))
end