using NetworkDynamics
using LightGraphs
using OrdinaryDiffEq
using Plots

### Defining the graph

N = 50 # nodes
k = 5 # node degree
g = barabasi_albert(N, k) # graph

### Network dynamics vertex and edge functions

@inline function kuramoto_vertex!(dv, v, e_s, e_d, p, t)
    # usually dv, v, e_s, e_d are arrays, hence we use the broadcasting operator .
    dv .= p
    oriented_edge_sum!(dv, e_s, e_d)
    nothing
end

@inline function kuramoto_edge!(e,v_s,v_d,p,t)
    # usually e, v_s, v_d are arrays, hence we use the broadcasting operator .
    e .= p * sin.(v_s .- v_d)
    nothing
end

### Constructing the network dynamics

odevertex = ODEVertex(f! = kuramoto_vertex!, dim = 1)
staticedge = StaticEdge(f! = kuramoto_edge!, dim = 1)

# generating random values for the parameter value ω_0 of the vertices
v_pars = [1. * randn() for v in vertices(g)]
# coupling stength of edges are set to 1/3
e_pars = [1. /3. for e in edges(g)]

parameters = (v_pars, e_pars)

# setting up the  network_dynamics
kuramoto_network! = network_dynamics(odevertex, staticedge, g)

### Simulation and Plotting

# constructing random initial conditions for nodes (variable θ)
x0 = randn(nv(g)) # nv(g) - number of vertices in g
dx = similar(x0)

prob = ODEProblem(kuramoto_network!, x0, (0.,2000), parameters)
sol = solve(prob, Tsit5())

plot(sol)

### Inspecting the Kuramoto order parameter

# s is a solution array
function order_parameter(s)
    θ  = 0
    for i in s
        θ += exp(im * i)
    end
    θ / length(s)
end

### Plotting

# first we get the indices of the vertex variables 'v'
u_idx = idx_containing(kuramoto_network!, :v)

# Then we compute the order parameter at each time step:
# sol[u_idx,:] is the multidimensional solution array
plot(abs.(mapslices(order_parameter, sol[u_idx, :], dims = 1))[1:1000])
