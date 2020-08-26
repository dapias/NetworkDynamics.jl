# Chapter 1

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
tspan = (0., 2.)
prob = ODEProblem(kuramoto_network, x0, tspan, ω)

sol = solve(prob, Tsit5())



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

# Chapter 2


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
vertex_array[1] = static!
vertex_array[5] = inertia!
nd_hetero!      = network_dynamics( vertex_array, edge!, g);

# Parameters and inital conditions

x0 = collect(1:(N+1)) ./ (N+1) .- .5
x0[6] = 3.
ω = collect(1:N)./N .- .5
ω[5] = ω[1]
ω[1] = 0.
ω[2:end] .-= sum(ω[2:end])/ (N-1)
x0[1] = ω[1] # initial conditions need to be corrected for the static vertex!
K  = 10.
p  = (ω, K)
tspan = (0., 30.)
prob_connected = ODEProblem(nd_hetero!, x0, tspan, p);
sol_connected = solve(prob_connected, Rosenbrock23());
vars = syms_containing(nd_hetero!, :θ);


# Hidden
membership = Int64.(ones(N))
membership[1] = 2
membership[5] = 3
nodecolor = [colorant"lightseagreen", colorant"orange", colorant"darkred"]
# membership color
nodefillc = nodecolor[membership]
nodefillc = reshape(nodefillc, 1, N)

plot(sol_connected, ylabel="θ", vars=vars, lc = nodefillc)
