using NetworkDynamics
using LightGraphs
using OrdinaryDiffEq
using DelayDiffEq
using Plots
using Distributions

t_max=1000
const c = 3.0
n=4
const coupling_constant = c/(n-1)
const q = 0.2

A=[0 1 0 0 ; 1 0 1 1; 0 1 0 0; 0 1 0 0]
g_undirected=SimpleGraph(A)
g_directed=SimpleDiGraph(A)

function kuramoto_vertex_undirected!(dv, v, e_s, e_d, p, t)
        dv[1] = p # parameter for vertices are eigenfrequencies omega
        @inbounds for e in e_s
            dv[1] -= e[1]
        end
        @inbounds for e in e_d
            dv[1] -= e[2]
        end
        nothing
end

function kuramoto_vertex_directed!(dv, v, e_s, e_d, p, t)
        dv[1] = p # parameter for vertices are eigenfrequencies omega
        @inbounds for e in e_s
            dv[1] -= e[1]
        end
        nothing
end

function kuramoto_vertex_undirected_improved!(dv, v, e_s, e_d, p, t)
        dv[1] = p # parameter for vertices are eigenfrequencies omega
        @inbounds for e in e_s
            dv[1] -= e[1]
        end
        @inbounds for e in e_d
            dv[1] += e[1]
        end
        nothing
end

function kuramoto_edge_undirected!(e,v_s,v_d,p,t)
        e[1] = p * sin(v_s[1] - v_d[1])
        e[2] = p * sin(v_d[1] - v_s[1])
    nothing
end

function kuramoto_edge_directed_and_improved!(e,v_s,v_d,p,t)
        e[1] = p * sin(v_s[1] - v_d[1])
    nothing
end


undirected_vertex = ODEVertex(f! = kuramoto_vertex_undirected!, dim = 1)
directed_vertex = ODEVertex(f! = kuramoto_vertex_directed!, dim = 1)
undirected_vertex_improved = ODEVertex(f! = kuramoto_vertex_undirected_improved!, dim = 1)

undirected_edge = StaticEdge(f! = kuramoto_edge_undirected!, dim = 2)
directed_edge = StaticEdge(f! = kuramoto_edge_directed_and_improved!, dim = 1)

v_pars = [rand(Uniform(-0.5, 0.5)) for v in vertices(g_undirected)]
e_pars_undirected = [coupling_constant for e in edges(g_undirected)]
e_pars_directed_and_improved = [coupling_constant for e in edges(g_directed)]

parameters_undirected = (v_pars, e_pars_undirected)
parameters_directed = (v_pars, e_pars_directed_and_improved)
# setting up the  network_dynamics
undirected_network! = network_dynamics(undirected_vertex, undirected_edge, g_undirected)
directed_network! = network_dynamics(directed_vertex, directed_edge, g_directed)
undirected_improved_network! = network_dynamics(undirected_vertex_improved, directed_edge, g_undirected)
### Simulation
# constructing random initial conditions for nodes (variable θ)
x0 = [rand(Uniform(0, 2*π)) for i in 1:nv(g_undirected)] # nv(g) - number of vertices in g

prob_undirected = ODEProblem(undirected_network!, x0, (0.,t_max), parameters_undirected, atol=1e-6, rtol=0)
prob_directed = ODEProblem(directed_network!, x0, (0.,t_max), parameters_directed, atol=1e-6, rtol=0)
prob_undirected_improved = ODEProblem(undirected_improved_network!, x0, (0.,t_max), parameters_directed, atol=1e-6, rtol=0)

### Benchmarking
time_to_solve_undirected = @elapsed(mod.(solve(prob_undirected, DP5(), tstops=0.:0.1:t_max, saveat=0.:0.1:t_max), 2π))
time_to_solve_directed   = @elapsed(mod.(solve(prob_directed, DP5(), tstops=0.:0.1:t_max, saveat=0.:0.1:t_max), 2π))
time_to_solve_undirected_improved = @elapsed(mod.(solve(prob_undirected_improved, DP5(), tstops=0.:0.1:t_max, saveat=0.:0.1:t_max), 2π))

sol_undirected = mod.(solve(prob_undirected, DP5(), tstops=0.:0.1:t_max, saveat=0.:0.1:t_max), 2π)
sol_directed = mod.(solve(prob_directed, DP5(), tstops=0.:0.1:t_max, saveat=0.:0.1:t_max), 2π)
sol_undirected_improved = mod.(solve(prob_undirected_improved, DP5(), tstops=0.:0.1:t_max, saveat=0.:0.1:t_max), 2π)

t = [1:10001]
plot_undirected = plot(t,sol_undirected[1,:], label = "Undirected Graph")

plot_undirected_and_directed = plot!(plot_undirected, t, sol_directed[1,:], label = "Directed Graph")
plot!(plot_undirected_and_directed, t, sol_undirected_improved[1,:], label = "Undirected and improved Graph")
