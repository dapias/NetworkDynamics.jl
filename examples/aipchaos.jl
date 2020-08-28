# Chapter 1 - Pure Julia

using LightGraphs, OrdinaryDiffEq

N = 10
g = watts_strogatz(N,  2, 0.)

const B = incidence_matrix(g, oriented=true)
const B_t = transpose(B)

function kuramoto_network(dθ, θ, ω, t)
    dθ .= ω .- (B * sin.(B_t * θ))
    return nothing
end

ω = (collect(1:N) .- sum(1:N) / N ) / N
x0 = (collect(1:N) .- sum(1:N) / N ) / N
tspan = (0., 10.)
prob = ODEProblem(kuramoto_network, x0, tspan, ω)

sol = solve(prob, Tsit5())


# Chapter 2 - Homogeneous ND

function kuramoto_edge!(e, θ_s, θ_d, σ, t)
    e .=  σ .* sin.(θ_s .- θ_d)
end
function kuramoto_vertex!(dθ, θ, e_s, e_d, ω, t)
    dθ .= ω
    oriented_edge_sum!(dθ, e_s, e_d)
end

using NetworkDynamics

vertex! = ODEVertex(f! = kuramoto_vertex!, dim = 1, sym=[:θ])
edge!   = StaticEdge(f! = kuramoto_edge!, dim = 1)
nd! = network_dynamics(vertex!, edge!, g)

vertexp = ω
edgep   = 5.
p = (vertexp, edgep)

nd_prob = ODEProblem(nd!, x0, tspan, p)
nd_sol = solve(nd_prob, Tsit5())

using Plots

plot(nd_sol, ylabel="θ")

# Chapter 3 - Heterogeneous ND

# Hidden
membership = Int64.(ones(N))
membership[1] = 2
membership[N ÷ 2] = 3
nodecolor = [colorant"lightseagreen", colorant"orange", colorant"darkred"];
# membership color
nodefillc = nodecolor[membership];
nodefillc = reshape(nodefillc, 1, N);

function kuramoto_inertia!(dθ, θ, e_s, e_d, p, t)
    dθ[1] = θ[2]
    dθ[2] = p -  θ[2]
    for e in e_s
        dθ[1]  -= e[1]
    end
    for e in e_d
        dθ[1] += e[1]
    end
end

inertia! = ODEVertex(f! = kuramoto_inertia!, dim = 2, sym= [:θ, :ω])

static! = StaticVertex(f! = (θ, e_s, e_d, c, t) -> θ .= c, dim = 1, sym = [:θ])


function kuramoto_edge!(e, θ_s, θ_d, K, t)
    e[1] = K * sin(θ_s[1] - θ_d[1])
end

vertex_array    = Array{VertexFunction}( [vertex! for i = 1:N])
vertex_array[1] = inertia!
vertex_array[N ÷ 2] = static!
nd_hetero!      = network_dynamics( vertex_array, edge!, g);

# Parameters and inital conditions

x0[N ÷ 2] = vertexp[N ÷ 2] # correct i.c. for static vertex
insert!(x0, 2, 3) # add initial condition for inertia vertex
p  = (vertexp, edgep)
prob_connected = ODEProblem(nd_hetero!, x0, tspan, p);
sol_connected = solve(prob_connected, Rosenbrock23());
vars = syms_containing(nd_hetero!, :θ);
plot(sol_connected, ylabel="θ", vars=vars, lc = nodefillc)





# Chapter 4 - Fancy ND (Delays)

using DelayDiffEq

function kuramoto_delay_edge!(e, v_s, v_d, h_v_s, h_v_d, p, t)
    # The coupling is no longer symmetric, so we need to store BOTH values (see tutorials for details)
    e[1] = p * sin(v_s[1] - h_v_d[1])
    e[2] = p * sin(h_v_s[1] - v_d[1])
    nothing
end
kdedge! = StaticDelayEdge(f! = kuramoto_delay_edge!, dim=2)



# redefine node functions to account for lost symmetry

function kuramoto_inertia!(dθ, θ, e_s, e_d, p, t)
    dθ[1] = θ[2]
    dθ[2] = p -  θ[2]
    for e in e_s
        dθ[1] -= e[1]
    end
    for e in e_d
        dθ[1] -= e[2]
    end
end

function kuramoto_vertex!(dθ, θ, e_s, e_d, ω, t)
    dθ .= ω
    for e in e_s
        dθ[1] -= e[1]
    end
    for e in e_d
        dθ[1] -= e[2]
    end
end

nd_delay! = network_dynamics(vertex_array, kdedge!, g)


h(out, p, t) = (out .= x0)
# p = (vertexparameters, edgeparameters, delaytime)
τ = 0.001
tspan = (0.,10.)
p = (vertexp,  edgep, τ)
dde_prob = DDEProblem(nd_delay!, x0, h, tspan, p)

sol_delay = solve(dde_prob, MethodOfSteps(Rosenbrock23(autodiff=false)));

plot(sol_delay, ylabel="θ", vars=vars, lc = nodefillc)

# Chapter  5 - Fancy ND II (Callbacks)
using DiffEqCallbacks

edgep_cut = ones(ne(g)) * 5.
edgep_cut[1] = 0

p_cut = (vertexp,  edgep_cut, τ)


condition(u,t,integrator) = t==5.
affect!(integrator) = integrator.p = p_cut
cb = DiscreteCallback(condition,affect!)
tspan = (0.,20.)
dde_prob = DDEProblem(nd_delay!, x0, h, tspan, p)

sol_delay = solve(dde_prob, MethodOfSteps(Rosenbrock23(autodiff=false)), callback=cb, tstops=[5.]);
plot(sol_delay, ylabel="θ", vars=vars, lc = nodefillc)
vline!([5.], color = [:black], width = [1.], line=[:dot])
