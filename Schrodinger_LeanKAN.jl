# PACKAGES AND INCLUSIONS
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, LinearAlgebra
using Random
using ModelingToolkit
using MAT
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using Random
using ForwardDiff
using Flux: ADAM, mae, update!, mean
using Flux
using Optimisers
using MethodOfLines

# DIRECTORY
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
fname       = "Schro_leankan"
add_path    = "results/"
mkpath(dir*add_path*"figs")
mkpath(dir*add_path*"checkpoints")

# KAN PACKAGE LOAD
include("src/KolmogorovArnold.jl")
using .KolmogorovArnold

# PARAMETERS, VARIABLES, AND DERIVATIVES
@parameters t x
@variables uʳ(..), uⁱ(..)
Dt      = Differential(t)
Dx      = Differential(x)
Dxx     = Differential(x)^2

# 1D PDE AND BCs
eqs     = [Dt(uʳ(t,x)) ~ 0.5*Dxx(uⁱ(t,x)) + (uʳ(t,x)^(2)+uⁱ(t,x)^(2))*uⁱ(t,x),
            Dt(uⁱ(t,x)) ~ -0.5*Dxx(uʳ(t,x)) - (uʳ(t,x)^(2)+uⁱ(t,x)^(2))*uʳ(t,x)]
xspan   = (-5.0,5.0)
tspan   = (0.0,pi/2)
dx      = 0.05
dt      = 0.01
xgrid   = xspan[1]:dx:xspan[2]
bcs     = [uʳ(0,x) ~ 2*sech(x),
            uⁱ(0,x) ~ 0.0,
            uʳ(t,xspan[1]) ~ uʳ(t,xspan[2]),
            uⁱ(t,xspan[1]) ~ uⁱ(t,xspan[2]),
            Dx(uʳ(t,xspan[1])) ~ Dx(uʳ(t,xspan[2])),
            Dx(uⁱ(t,xspan[1])) ~ Dx(uⁱ(t,xspan[2]))]

# SPACE AND TIME DOMAINS
domains = [t ∈ IntervalDomain(tspan[1],tspan[2]),
           x ∈ IntervalDomain(xspan[1],xspan[2])]

# PDE SYSTEM
@named pdesys   = PDESystem(eqs,bcs,domains,[t,x],[uʳ(t,x), uⁱ(t,x)])

# METHOD OF LINES DISCRETIZATION
order           = 2
discretization  = MOLFiniteDifference([x => dx], t, approx_order=order)

# GENERATE TRAINING DATA
prob        = discretize(pdesys,discretization)
u0          = [0.0; prob.u0[1:length(xgrid)-1]; 0.0; prob.u0[length(xgrid):end]]
sol         = solve(prob,Rodas5(),saveat=dt)
xgrid       = xspan[1]:dx:xspan[2]
tgrid       = tspan[1]:dt:tspan[2]
idx_        = [11, 31, 51, 71, 91, 111, 131, 151]
dt_train    = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
Xₙʳ         = zeros(length(idx_),length(xgrid))
Xₙⁱ         = zeros(length(idx_),length(xgrid))
for (i, idx) ∈ enumerate(idx_)
    Xₙʳ[i,:]    = sol.u[uʳ(t,x)][idx,:]
    Xₙⁱ[i,:]    = sol.u[uⁱ(t,x)][idx,:]
end
Xₙ      = [Xₙʳ'; Xₙⁱ']'

# PLOT TRAINING DATA
pythonplot()
contourf(sol.t, xgrid, sqrt.(sol.u[uʳ(t,x)].^(2).+sol.u[uⁱ(t,x)].^(2))', color=:turbo, levels=201, size=(600, 250))
xlabel!("t")
ylabel!("x")

# DEFINE THE NETWORK OF KAN-ODEs
basis_func  = rbf
normalizer  = softsign
KANgrid     = 5;
KANnode     = 3;
kan1 = Lux.Chain(
    KDense(size(xgrid)[1]*2, KANnode, KANgrid; use_base_act = true, basis_func, normalizer),
    KDense_lean(KANnode, size(xgrid)[1]*2, KANgrid; use_base_act = true, basis_func, normalizer, mult_flag=2),
)
rng         = Random.default_rng()
Random.seed!(rng, 0)
pM , stM    = Lux.setup(rng, kan1)
pM_data     = getdata(ComponentArray(pM))
pM_axis     = getaxes(ComponentArray(pM))

# CONSTRUCT KAN-ODES
train_node  = NeuralODE(kan1, tspan, Tsit5(), saveat = dt_train)

# PREDICTION FUNCTION
function predict(p)
    Array(train_node(u0, p, stM)[1])
end

# LOSS FUNCTION
function loss(p)
    mean(abs2, Xₙ .- predict(ComponentArray(p,pM_axis))')
end

# CALLBACK FUNCTION
function callback(i)
    if i%500 == 0
        # SAVE PARAMETERS AND LOSS
        # p_list in mat form
        # loss in mat form
        l_ = zeros(size(l))
        for j = 1:size(l,1)
            l_[j] = l[j]
        end
        file = matopen(dir*add_path*"/checkpoints/"*fname*"_results.mat", "w")
        write(file, "p_opt", p_opt)
        write(file, "loss", l_)
        close(file)


        p0 = plot(l, yaxis=:log, label=:none)
        xlabel!("Epoch")
        ylabel!("Loss")

        tspan_test = tspan
        tgrid_test = tspan_test[1]:dt:tspan_test[2]
        prob_ = remake(prob, tspan = tspan_test)
        sol_true = solve(prob_,Rodas5(),saveat=dt)
        clims_re=minimum(sol_true.u[uʳ(t,x)]), maximum(sol_true.u[uʳ(t,x)])
        clims_im=minimum(sol_true.u[uⁱ(t,x)]), maximum(sol_true.u[uⁱ(t,x)])
        p1 = contourf(sol_true.t, xgrid, sol_true.u[uʳ(t,x)]', title="True Re(u)", xlabel="x", ylabel="t", color=:Reds_8, levels=201, size=(600, 250), clims= clims_re)
        p2 = contourf(sol_true.t, xgrid, sol_true.u[uⁱ(t,x)]', title="True Im(u)", xlabel="x", ylabel="t", color=:Reds_8, levels=201, size=(600, 250), clims=clims_im)
        

        train_node_ = NeuralODE(kan1, tspan_test, Tsit5(), saveat = dt)
        pred_sol_lean = train_node_(u0, ComponentArray(p,pM_axis), stM)[1]
        pred_sol_lean_array = Array(pred_sol_lean)
        p3 = contourf(pred_sol_lean.t, xgrid, pred_sol_lean_array[1:length(xgrid),:], title="LeanKAN Re(u)", xlabel="x", ylabel="t", color=:Reds_8, levels=201, size=(600, 250), clims= clims_re)
        p4 = contourf(pred_sol_lean.t, xgrid, pred_sol_lean_array[length(xgrid)+1:end,:], title="LeanKAN Im(u)", xlabel="x", ylabel="t", color=:Reds_8, levels=201, size=(600, 250), clims=clims_im)
        
        lay = @layout [a{0.35h}
        [grid(2, 2) ]]
        pt = plot(p0, p1, p2, p3, p4, layout=lay)
        savefig(pt, dir*add_path*"/figs/"*fname*"_result.png")
    end
end


# TRAINING SETUP
isrestart   = false
p           = deepcopy(pM_data)
p_opt=p
min_loss=1
opt         = Flux.Adam(5e-4)
l           = []
N_iter      = 1e5
i_current   = 1
append!(l, [deepcopy(loss(p))])

if isrestart == true
    file        = matopen(dir*add_path*"/checkpoints/"*fname*"_results.mat")
    p_opt     = read(file, "p_opt")
    l_          = read(file, "loss")
    close(file)
    i_current   = length(l_)
    min_loss=loss(p_opt)
    for i = 1:i_current
        append!(l, [l_[i]])
    end

    p = p_opt

    l_          = nothing
end

# TRAINING LOOP
using Zygote
for i = i_current:N_iter
    global i_current
    global min_loss
    global p_opt

    
    # GRADIENT COMPUTATION
    grad    = Zygote.gradient(x->loss(x), p)[1]

    # UPDATE WITH ADAM OPTIMIZER
    update!(opt, p, grad)

    # PARAM, LOSS
    append!(l, [deepcopy(loss(p))])
    if loss(p)<min_loss
        p_opt=deepcopy(p)
        min_loss=loss(p)
    end
    # CALLBACK
    println("Iteration: ", Int32(i_current),"/", Int32(N_iter), ", Loss: ", l[Int32(i)])
    i_current = i_current + 1

    # SAVE
    callback(i)
end