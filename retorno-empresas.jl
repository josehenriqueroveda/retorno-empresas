using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
using Clustering
using Distances

# UNSUPERVISED MACHINE LEARNING - CLUSTERING
# Carregamento da base de dados

function dados_empresas()
    path = "empresas.csv"
    df = CSV.read(path, DataFrame)
    select(df, Not([:Column1]))
end

df_empresas = dados_empresas()

# Primeiros 5 registros dos dados
first(df_empresas, 5)

# Descrição do dataset
@show describe(df_empresas)


# Visualização dos dados - Correlação
@df df_empresas corrplot(cols(2:6))

cor(Matrix(df_empresas[:, 2:6]'))

# K-Means Clustering
X = df_empresas[!, 2:6]
C = kmeans(Matrix(X)', 3)

insertcols!(df_empresas, 7, :cluster3 => C.assignments)

scatter(df_empresas.liquidez, df_empresas.retorno, marker_z=C.assignments,
        color=:lightrainbow, legend=true)

# K-Medoids Clustering
xmatrix = Matrix(X)'
D = pairwise(Euclidean(), xmatrix, xmatrix, dims = 2)
K = kmedoids(D, 3)

insertcols!(df_empresas, 8, :medoids_cluster => K.assignments)

scatter(df_empresas.liquidez, df_empresas.retorno, marker_z=K.assignments,
        color=:lightrainbow, legend=true)

# Agrupamento Hierárquico
K = hclust(D)
L = cutree(K, k=3)

insertcols!(df_empresas, 9, :hclusters => L)

