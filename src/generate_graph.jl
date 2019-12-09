using Plots, LightGraphs, CSV, LinearAlgebra, VoronoiDelaunay, JLD

include("F://GitHub//Graph-Convolutional-Neural-Network//src//gplot.jl")
include("F://GitHub//Graph-Convolutional-Neural-Network//src//helpers.jl")

df = CSV.read("F://GitHub//Graph-Convolutional-Neural-Network//dataset//singapore_airbnb.csv")

latitude = df.latitude[:]
longitude = df.longitude[:]
price = df.price[:]; price[price .> 250] .= 250

X = hcat(longitude, latitude)




# scatter_gplot(X; marker = price, ms = 3)

N_raw = size(X,1)
remove_nodes = []
dist = zeros(N_raw,N_raw); for i = 1:N_raw-1, j = i+1:N_raw; dist[i,j] = norm(X[i,:] - X[j,:]); if dist[i,j] < 1e-10; push!(remove_nodes, [i,j]);end; end; dist = dist + dist';

N = N_raw - length(remove_nodes)
rest_idx = setdiff(1:N_raw, [remove_nodes[k][2] for k in 1:length(remove_nodes)])
X = X[rest_idx,:]
dist = dist[rest_idx, rest_idx]


tess = DelaunayTessellation(N)
m1 = minimum(X[:,1]); M1 = maximum(X[:,1]); width1 = M1 - m1; m2 = minimum(X[:,2]); M2 = maximum(X[:,2]); width2 = M2 - m2; width = max_coord - min_coord
a = Point2D[Point(min_coord + ((X[i,1] - m1) / width1) * width, min_coord + ((X[i,2] - m2) / width2) * width) for i in 1:N]
push!(tess, a)


h = Dict(); for i in 1:N; h[[min_coord + ((X[i,1] - m1) / width1) * width, min_coord + ((X[i,2] - m2) / width2) * width]] = i; end

# x, y = getplotxy(delaunayedges(tess))
# plot(x,y,linestyle=:auto, aspect_ratio = 1, legend = false)

G = Graph(N)
for edge in delaunayedges(tess)
    idx1 = h[[getx(geta(edge)), gety(geta(edge))]]
    idx2 = h[[getx(getb(edge)), gety(getb(edge))]]
    add_edge!(G,Edge(idx1,idx2))
end

# savegraph("F:\\GitHub\\Graph-Convolutional-Neural-Network\\dataset\\delunary_singapore_graph.lgz", G)
# JLD.save("F:\\GitHub\\Graph-Convolutional-Neural-Network\\dataset\\delunary_singapore_XnIdx.jld", "X", X, "rest_idx", rest_idx)

# ## build k-nearest neighbor graph, k = 3,5
# G = Graph(N)
# for v = 1:N
#     dist2v = dist[v,:]
#     for nb in findminimum(dist[v,:], 3)
#         add_edge!(G,Edge(v,nb))
#     end
# end

gr(dpi = 400); gplot(1.0 .* adjacency_matrix(G), X; width = 1); plt = plot!(aspect_ratio = 1)
# savefig(plt, "F:\\GitHub\\Graph-Convolutional-Neural-Network\\figs\\delaunay_graph.png")


# x, y = getplotxy(delaunayedges(tess))
# plot(x,y,linestyle=:auto, aspect_ratio = 1, legend = false)
