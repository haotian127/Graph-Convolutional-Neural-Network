using Plots, LightGraphs, Arpack

G = loadgraph("F:\\GitHub\\Graph-Convolutional-Neural-Network\\dataset\\delunary_singapore_graph.lgz")
@load "F:\\GitHub\\Graph-Convolutional-Neural-Network\\dataset\\delunary_singapore_XnIdx.jld"

N = nv(G)
A = 1.0 .* adjacency_matrix(G)
L = 1.0 .* laplacian_matrix(G)
λ, ϕ = eigs(L, nev = 10, which=:SM)

scatter_gplot(X; marker = ϕ[:,7])

scatter_gplot(ϕ[:,3:4]; marker = ϕ[:,2])
