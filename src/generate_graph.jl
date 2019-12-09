using Plots, LightGraphs, CSV, LinearAlgebra, VoronoiDelaunay

include("F://GitHub//Graph-Convolutional-Neural-Network//src//gplot.jl")
include("F://GitHub//Graph-Convolutional-Neural-Network//src//helpers.jl")

df = CSV.read("F://GitHub//Graph-Convolutional-Neural-Network//dataset//singapore_airbnb.csv")

latitude = df.latitude[:]
longitude = df.longitude[:]
price = df.price[:]; price[price .> 250] .= 250

X = hcat(longitude, latitude)

# scatter_gplot(X; marker = price, ms = 3)

N = size(X,1)
# dist = zeros(N,N); for i = 1:N-1, j = i+1:N; dist[i,j] = norm(X[i,:] - X[j,:]); end; dist = dist + dist';

tess = DelaunayTessellation()
a = Point2D[Point(X[i,1] - 102, X[i,2]) for i in 1:N]
push!(tess, a)

h = Dict(); for i in 1:N; h[X[i,:]] = i; end

# x, y = getplotxy(delaunayedges(tess))
# plot(x,y,linestyle=:auto, aspect_ratio = 1, legend = false)

G = Graph(N)
for edge in delaunayedges(tess)
    idx1 = h[[getx(geta(edge))+102, gety(geta(edge))]]
    idx2 = h[[getx(getb(edge))+102, gety(getb(edge))]]
    add_edge!(G,Edge(idx1,idx2))
end
savegraph(G, "dataset\\delunary_singapore_graph.lgz")

# ## build k-nearest neighbor graph, k = 3,5
# G = Graph(N)
# for v = 1:N
#     dist2v = dist[v,:]
#     for nb in findminimum(dist[v,:], 3)
#         add_edge!(G,Edge(v,nb))
#     end
# end

gplot(1.0 .* adjacency_matrix(G), X; width = 1); plt = plot!(aspect_ratio = 1)
savefig(plt, "figs\\delaunay_graph.png")
