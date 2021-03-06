using DifferentialEquations
using NetworkDynamics
using NLsolve
using LightGraphs
using Plots
using GraphPlot
using Revise

g = star_graph(4)
gplot(g)

function swing_eq!(dv, v, e_s, e_d, P, t)
    dv[1] = v[2]
    dv[2] = P - 0.1 * v[2]
    for e in e_s
        dv[2] -= e[1]
    end
    for e in e_d
        dv[2] += e[1]
    end
end

swing_vertex = ODEVertex(f! = swing_eq!, dim = 2, sym=[:θ, :ω])

function load_eq!(dv, v, e_s, e_d, P, t)
    dv[1] = P
    oriented_symmetric_edge_sum!(dv, e_s, e_d)
end

load_vertex    = ODEVertex(f! = load_eq!, dim = 1, mass_matrix = 0, sym=[:θ])

function powerflow_eq!(e, v_s, v_d, K, t)
    e[1] = K * sin(v_s[1] - v_d[1])
end

powerflow_edge = StaticEdge(f! = powerflow_eq!, dim = 1)

vertex_array = [swing_vertex, swing_vertex, load_vertex, load_vertex]
edge_array = [powerflow_edge for e in edges(g)]
nd = network_dynamics(vertex_array, edge_array, g)

K = 6.0
P = [1.0, 1.0, -1.0, -1.0]
p = (P,K)

u0 = find_fixpoint(nd, p, zeros(6))
tspan = (0., 100.)
ode_prob = ODEProblem(nd, u0, tspan, p)
ode_sol = solve(ode_prob, Rosenbrock23())
plot(ode_sol, vars = syms_containing(nd, "θ"), legend = false)
