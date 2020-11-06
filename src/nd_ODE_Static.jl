module nd_ODE_Static_mod

using ..NetworkStructures
using ..ComponentFunctions
using ..Utilities


export nd_ODE_Static
export StaticEdgeFunction


@inline function prep_gd(dx::T, x::T, gd::GraphData{T, T}, gs) where T
    # Type matching
    gd.v_array = x # Does not allocate
    gd
end

@inline function prep_gd(dx, x, gd, gs)
    # Type mismatch
    e_array = similar(dx, gs.dim_e)
    GraphData(x, e_array, gs)
end



@Base.kwdef struct nd_ODE_Static{G, Tv, Te, T1, T2}
    vertices!::T1
    edges!::T2
    graph::G #redundant?
    graph_structure::GraphStruct
    graph_data::GraphData{Tv, Te}
    parallel::Bool # enables multithreading for the core loop
end


function (d::nd_ODE_Static)(dx, x, p, t)
    gd = prep_gd(dx, x, d.graph_data, d.graph_structure)
    gs = d.graph_structure

    @nd_threads d.parallel for i in 1:gs.num_e
        maybe_idx(d.edges!, i).f!(view(gd.e_array, gs.e_idx[i]),
                                  view(x, gs.s_e_idx[i]),
                                  view(x, gs.d_e_idx[i]),
                                  p_e_idx(p, i),
                                  t)
    end

    @nd_threads d.parallel for i in 1:gs.num_v
        maybe_idx(d.vertices!,i).f!(view(dx,gs.v_idx[i]),
                                    view(x, gs.v_idx[i]),
                                    view(gd.e_views, gs.e_s_v_idx[i]),
                                    view(gd.e_views, gs.e_d_v_idx[i]),
                                    p_v_idx(p, i), t)
    end

    nothing
end


function (d::nd_ODE_Static)(x, p, t, ::Type{GetGD})
    gd = prep_gd(x, x, d.graph_data, d.graph_structure)


    @nd_threads d.parallel for i in 1:d.graph_structure.num_e
        maybe_idx(d.edges!,i).f!(gd.e[i], gd.v_s_e[i], gd.v_d_e[i], p_e_idx(p, i), t)
    end

    gd
end


function (d::nd_ODE_Static)(::Type{GetGS})
    d.graph_structure
end


# For compatibility with PowerDynamics

struct StaticEdgeFunction
    nd_ODE_Static
end

function (sef::StaticEdgeFunction)(x, p, t)
    gd = sef.nd_ODE_Static(x, p, t, GetGD)

    (gd.e_s_v, gd.e_d_v)
end

end #module
