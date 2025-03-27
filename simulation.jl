using Statistics
using Distributions
using DataFrames
using Random
using Plots
using Polynomials
using Optim
using NamedArrays
using JLD2


#Supply side

argmax_df(df, arg, value) = df[df[!,value] .== maximum(df[!,value]), arg]

function bound_price(x)
    if x < c
        return c
    elseif x > p_upper
        return p_upper
    else
        return x
    end
end

function calc_prices(Qs, ε, action_set, method, σ_p)
    if method == :ε_greedy
        [rand() < ε ? rand(action_set) : rand(argmax_df(Qs[i], :p, :q)) for i in 1:N]
    elseif method == :ε_local
        [rand() < ε ? bound_price(round(rand(Normal(rand(argmax_df(Qs[i], :p, :q)), σ_p)), digits =1)) : rand(argmax_df(Qs[i], :p, :q)) for i in 1:N]
    else
        error("Method not recognized")
    end
end

#calc_prices(Qs, ε, action_set) = [rand() < ε ? rand(action_set) : rand(argmax_df(Qs[i], :p, :q)) for i in 1:N]

profit(M, s, p, c) = M * s * (p - c)

function update(Q, M, s, p, c, μ, γ)
    r = profit(M, s, p, c)
    q = Q.q[Q.p .== p][1]
    r,  q + μ * (r +  γ * maximum(Q.q) - q)
end 


#shares(demand_params, [25.1], [1])
#profit(M, 0.99, 25, c)



#Demand side




function nums(demand_params, P_T, bb_T)
    ξ, α, β = demand_params
    N = length(P_T)
    exp.(hcat(fill(1, N), P_T, bb_T) * transpose(hcat(fill(ξ,M), α, β)))

end


denom(nummerators) = transpose(1 .+ sum.(eachcol(nummerators)))


function shares(demand_params, P_T, bb_T)
    nummerators = nums(demand_params, P_T, bb_T)
    denomerators = denom(nummerators)
    mean.(eachrow(nummerators ./ denomerators))
end





#Platform

    #P = [2 1 1 4 5; 2 5 1 4 5; 2 10 5 2 5]

    #bb::BitArray = [0 0 0 1 0; 0 0 1 0 0]



    # function buyboxWinner1(P)
    #     P_ind = P .== minimum(P)

    #     if sum(P_ind) > 1
    #         sellers = findall(P_ind .== 1)
    #         buybox = repeat([0], N)
    #         buybox[rand(sellers)] = 1
    #     else
    #         buybox = P .== minimum(P)
    #     end
    #     buybox
    # end

function randomizer(vector::BitVector)
    new_vector::BitVector = repeat([0], length(vector))
    ind = findall(vector)
    new_vector[rand(ind)] = 1
    new_vector
end

function buyboxWinner(P, bb, δ)
    N = size(P,2)
    T = size(P,1)
    P_T = P[T,:]
    minP = minimum(P_T)
    is_currentPeriodLP = P_T .== minP

    if T == 1
        buybox = randomizer(is_currentPeriodLP) # pick a random seller from the ones with the lowest price
    elseif T != size(bb,1) + 1
        error("'P' must be exactly one row longer than 'bb'")
    else
        
        bb_T_1 = bb[T-1,:]
        p_T_1_winner_T_1 = P[T-1,bb_T_1][1]
        p_T_winner_T_1 = P[T,bb_T_1][1]
        
        #WP stands for "Winner Price"
        
        #minP_T_1 = minimum(P[T-1,:])
        
        #i_lastPeriodWinner = findall(bb_T_1)
        #i_currentPeriodLP = findall(is_currentPeriodLP)
        #if i_lastPeriodWinner ∈ i_currentPeriodLP
        #    buybox = bb_T_1
        if minP < p_T_1_winner_T_1 * (1 - δ)
            buybox = randomizer(is_currentPeriodLP)
        elseif minP < p_T_winner_T_1 * (1 - δ)
            buybox = randomizer(is_currentPeriodLP)
        else
            buybox = bb_T_1
        end

    end
    
    buybox
end



function p_deviation(Q, m_price)
    max_q = maximum(Q.q)
    mean(abs.(Q.p .- m_price) .* Q.q ./ max_q)
end



function simulate(params, T, method, add_plot = false)

    N, M, c, μ, ε, γ, action_set, ξ, α, β, σ_p, δ = params


    #α = rand(Normal(α_bar, σ_α), M)
    #β = rand(Normal(β_bar, σ_β), M)

    demand_params = ξ, α, β

    Q = DataFrame(p = action_set, q = fill(0.0, length(action_set)))
    #Q = Dict(action_set .=> fill(0.0, length(action_set)))
    Qs = Dict(i => copy(Q) for i in 1:N)

    # Step 1
    P_T::Vector{Float64} = calc_prices(Qs, ε, action_set, method,σ_p)
    P = transpose(P_T)

    # Step 2
    bb = []
    bb_T::BitVector = buyboxWinner(P, bb, δ)
    bb = transpose(bb_T)

    # Step 3
    s_T::Vector{Float64}  = shares(demand_params, P_T, bb_T)
    s = transpose(s_T)

    # Step 4
    r_T = fill(0.0, N)
    hashem_T = fill(0.0, N)
    
    for ii in 1:N 
        r_temp::Float64, q_temp::Float64 = update(Qs[ii], M, s_T[ii], P_T[ii], c, μ, γ)
        r_T[ii] = r_temp

        #println(ii, "_", P_T[ii], "_", Qs[ii].q[findall(Qs[ii].p .== P_T[ii])[1]])
        Qs[ii].q[findall(Qs[ii].p .== P_T[ii])[1]] = q_temp
        #println(ii, "_", P_T[ii], "_", Qs[ii].q[findall(Qs[ii].p .== P_T[ii])[1]])
        hashem_T[ii] = p_deviation(Qs[ii], m_price)
        
    end

    r = transpose(r_T)
    hashem = transpose(hashem_T)


    anim_temp1 = []
    anim_temp2 = []
    for tt in 1:T 
         P, bb, s, r, Qs, hashem = nextround(Qs, ε, P, bb, δ, demand_params, N, M, c, μ, γ, s, r, σ_p, hashem, method)
        if add_plot == true
            if tt % 1 == 0
                
                #Qs1 = Qs[1] # Q-values for seller 1
                #Qs2 = Qs[2] # Q-values for seller 2
                plott1 = gen_plot(Qs, 1, tt, 30000)
                plott2 = gen_plot(Qs, 2, tt, 30000)
                #plott = scatter(Qs1.p, Qs1.q, markershape = :xcross, markersize = 2, markercolor = :darkblue, ylimits = (0, 30000), legend = false, title = "Firm 1", size = (500, 500), xlabel = "Price", ylabel = "Q-value")
                #vline!([21.3], color = :green, label = "Monopoly Price")
                append!(anim_temp1, plott1)
                append!(anim_temp2, plott2)
                #Qs1 = Qs[1] # Q-values for seller 1
                #append!(anim_temp, scatter(Qs1.p, Qs1.q, markershape = :xcross, markersize = 2, markercolor = :darkblue, ylimits = (0, 80000)))


            end
        end
    end
    if add_plot == true
        anim1 = @animate for ii in 1:length(anim_temp1)
            plot(anim_temp1[ii])
        end
        anim2 = @animate for ii in 1:length(anim_temp2)
            plot(anim_temp2[ii])
        end
    else
        anim1 = []
        anim2 = []
    end
    
    P, bb, s, r, Qs, hashem, anim1, anim2
end








function nextround(Qs, ε, P, bb, δ, demand_params, N, M, c, μ, γ, s, r, σ_p, hashem, method)
    
    # Step 1
    P_T::Vector{Float64} = calc_prices(Qs, ε,action_set, method, σ_p)
    P = vcat(P, transpose(P_T))

    # Step 2
    bb_T::BitVector = buyboxWinner(P, bb, δ)
    bb = vcat(bb, transpose(bb_T))

    # Step 3
    s_T::Vector{Float64} = shares(demand_params, P_T, bb_T)
    s = vcat(s, transpose(s_T))

    # Step 4
    r_T = fill(0.0, N)
    hashem_T = fill(0.0, N)
    
    for ii in 1:N 

        r_temp::Float64, q_temp::Float64 = update(Qs[ii], M, s_T[ii], P_T[ii], c, μ, γ)
        r_T[ii] = r_temp
        #println(ii, "_", P_T[ii], "_", Qs[ii].q[findall(Qs[ii].p .== P_T[ii])[1]])
        Qs[ii].q[findall(Qs[ii].p .== P_T[ii])[1]] = q_temp
        #println(ii, "_", P_T[ii], "_", Qs[ii].q[findall(Qs[ii].p .== P_T[ii])[1]])
        hashem_T[ii] = p_deviation(Qs[ii], m_price)
    end

    r = vcat(r, transpose(r_T))
    hashem = vcat(hashem, transpose(hashem_T))

    P, bb, s, r, Qs, hashem
    
end

# Plot funcs

function gen_plot2()
    

    plot(scatter(Qs1.p, Qs1.q, markershape = :xcross, markersize = 2, markercolor = :darkblue, ylimits = (0, 30000), legend = false, title = "Firm 1"),
        scatter(Qs2.p, Qs2.q, markershape = :xcross, markersize = 2, markercolor = :darkred, ylimits = (0, 30000), legend = false, title = "Firm 2"),
            layout = (1,2),
            size = (1000, 500),
            xlabel = "Price",
            ylabel = "Q-value",
            margin = 0.5Plots.cm)

    vline!([21.3 21.3], color = :green, label = "Monopoly Price")
end

function gen_plot(Qs, firm, t, height)
    Q = Qs[firm] # Q-values for seller 1
    scatter(Q.p, Q.q, markershape = :xcross, markersize = 2, markercolor = :darkblue, ylimits = (0, height), legend = false, title = "Firm $firm Q at t=$t", size = (500, 500), xlabel = "Price", ylabel = "Q-value")
    vline!([21.3], color = :green, label = "Monopoly Price")
end


#Supply params

N = 2 # number of sellers

c = 10 # marginal cost
μ = 0.9 # learning rate # 0.9
ε = 0.8 # exploration rate # 0.8

σ_p = 1.8 # standard deviation of the price distribution

γ = 0.7 #discount factor

p_upper = 40

action_set = c:0.1:p_upper # price set

# Damand params

M = 1000 # number of buyers

ξ = 100 
α_bar = -5 
σ_α = 1.5
β_bar = 30
σ_β = 10

Random.seed!(7)

    α = rand(Normal(α_bar, σ_α), M)
    β = rand(Normal(β_bar, σ_β), M)

#Platform params

δ = 0.1

# Other params

m_price = 21.3


params = N, M, c, μ, ε, γ, action_set, ξ, α, β, σ_p, δ

T =1000 # number of periods





#####################



@time P, bb, s, r, Qs, hashem, anim1, anim2 = simulate(params, T, :ε_greedy, true)

#@code_warntype simulate(params, T, :ε_local, false)

#gen_plot(Qs, 2, T, 30000)

gif(anim1, "firm1_greedy.gif", fps = 120)
gif(anim2, "firm2_greedy.gif", fps = 120)

firm1_local = anim1
firm2_local = anim2

firm1_greedy = anim1
firm2_greedy = anim2

#anim = @animate for tt in 1:T
#    vcat(
#        hcat(load("$(firm1_greedy.dir)/$(firm1_greedy.frames[tt])"), load("$(firm2_greedy.dir)/$(firm2_greedy.frames[tt])")),
#        hcat(load("$(firm1_local.dir)/$(firm1_local.frames[tt])"), load("$(firm2_local.dir)/$(firm2_local.frames[tt])"))
#    )
#end

#gif(anim, "Q_evolution.gif", fps = 120)
anim = animate([vcat(
           hcat(load("$(firm1_greedy.dir)/$(firm1_greedy.frames[tt])"), load("$(firm2_greedy.dir)/$(firm2_greedy.frames[tt])")),
            hcat(load("$(firm1_local.dir)/$(firm1_local.frames[tt])"), load("$(firm2_local.dir)/$(firm2_local.frames[tt])"))
        ) for tt in 1:T], size = (1600, 1000), showaxis = false, fps = 120)







# Profits of bb and non-bb sellers
profs = DataFrame(  t = 1:T+1,
                    bb_ID = ifelse.(bb[:,1] .== 1, 1, 2) ,
                    bb_p = vec(sum(P .* bb, dims = 2)), # summing across rows
                    bb_profs = vec(sum(r .* bb, dims = 2)), 
                    non_bb_p = vec(sum(P .* (1 .- bb), dims = 2)),
                    non_bb_profs = vec(sum( r .* (1 .- bb), dims = 2)), 
                    total_profs = vec(sum(r, dims = 2))) 

sum(profs.total_profs)

function difference_prof_greedy_close(n_runs, params)
    prof_greedy = []
    prof_close = []
    for ii in 1:n_runs
        P, bb, s, r, Qs, hashem, anim = simulate(params, T, :ε_greedy, false)
        profs = DataFrame(  t = 1:T+1,
                        bb_ID = ifelse.(bb[:,1] .== 1, 1, 2) ,
                        bb_p = vec(sum(P .* bb, dims = 2)), # summing across rows
                        bb_profs = vec(sum(r .* bb, dims = 2)), 
                        non_bb_p = vec(sum(P .* (1 .- bb), dims = 2)),
                        non_bb_profs = vec(sum( r .* (1 .- bb), dims = 2)), 
                        total_profs = vec(sum(r, dims = 2))) 

        prof_greedy = vcat(prof_greedy, sum(profs.total_profs))

        P, bb, s, r, Qs, hashem, anim = simulate(params, T, :ε_local, false)
        profs = DataFrame(  t = 1:T+1,
                        bb_ID = ifelse.(bb[:,1] .== 1, 1, 2) ,
                        bb_p = vec(sum(P .* bb, dims = 2)), # summing across rows
                        bb_profs = vec(sum(r .* bb, dims = 2)), 
                        non_bb_p = vec(sum(P .* (1 .- bb), dims = 2)),
                        non_bb_profs = vec(sum( r .* (1 .- bb), dims = 2)), 
                        total_profs = vec(sum(r, dims = 2))) 

        prof_close = vcat(prof_close, sum(profs.total_profs))

    

    end
    mean(prof_close) - mean(prof_greedy)
end

@time difference_prof_greedy_close(2000, params)



σ_p = collect(0.2:0.2:8.0)

δ = collect(0.0:0.01:0.39)



function fill_grid(n_runs)
    grid = NamedArray(zeros(length(σ_p), length(δ)), (σ_p, δ), ("σ_p", "δ"))
    Threads.@threads for σ_p_ii in σ_p
        for δ_ii in δ
            params = N, M, c, μ, ε, γ, action_set, ξ, α, β, σ_p_ii, δ_ii
            grid[σ_p_ii, δ_ii] = difference_prof_greedy_close(n_runs, params)
        end
    end
    grid
end

@time results_grid = fill_grid(1000)

save_object("results_grid.jld2", results_grid)

color_gradient = cgrad([:red, :white, :blue], [0, 0.42, 1])

x_labels = names(results_grid, 2)  # Column names (δ)
y_labels = names(results_grid, 1)  # Row names (σ_p)

heatmap(x_labels, y_labels, results_grid, xlabel="δ", ylabel="σ_p", title="Total Cumulative Profit at T=100 (Localized vs. Regular)", color = color_gradient, left_margin = 0.8Plots.cm, right_margin = 0.8Plots.cm, size = (750, 650))

savefig("profit_heatmap.png")




# Fit a polynomial to the data
#fitted  = Polynomials.fit(1.0:T+1.0, profs.bb_profs, 20)

scatter(r[:,1], markershape = :xcross, markersize = 2, label = "Seller 1")

scatter(profs.total_profs, markershape = :xcross, markersize = 2)#; plot!(fitted, extrema(1:T+1)..., label = "Best Fit Curve", linestyle = :dash)

scatter(profs.bb_p, markershape = :xcross, markersize = 2)















