using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
using Clustering
using Distances
using GLM

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

cor(Matrix(df_empresas[:, 2:6]))


################################################################################################
#                       UNSUPERVISED MACHINE LEARNING - CLUSTERING                             #
################################################################################################

# K-Means Clustering
X = df_empresas[!, 2:6]
C = kmeans(Matrix(X)', 3)

insertcols!(df_empresas, 7, :k_cluster => C.assignments)

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

plot(K)


################################################################################################
#                   SUPERVISED MACHINE LEARNING - LINEAR REGRESSION                            #
################################################################################################

df_empresas = dados_empresas()

histogram(df_empresas.retorno, bins = 20)
density(df_empresas.retorno, )

println("Correlação entre Ativos e Retorno:", cor(df_empresas.retorno, df_empresas.ativos), "\n\n")
s1 = scatter(df_empresas.retorno, df_empresas.ativos, title = "Relação entre Ativos e Retorno",
                ylabel = "Ativos", xlabel = "Retorno", legend = false)


println("Correlação entre Liquidez e Retorno:", cor(df_empresas.retorno, df_empresas.liquidez), "\n\n")
s2 = scatter(df_empresas.retorno, df_empresas.liquidez, title = "Relação entre Liquidez e Retorno",
ylabel = "Liquidez", xlabel = "Retorno", legend = false)


println("Correlação entre Disclosure e Retorno:", cor(df_empresas.retorno, df_empresas.disclosure), "\n\n")
s3 = scatter(df_empresas.retorno, df_empresas.liquidez, title = "Relação entre Disclosure e Retorno",
ylabel = "Disclosure", xlabel = "Retorno", legend = false)

plot(s1, s2, s3)

# Regressão Linear - Retorno ~ Ativos
lm1 = lm(@formula(retorno ~ ativos), df_empresas[!, 2:6])
r2(lm1.model)
# Regressão Linear - Retorno ~ Liquidez
lm2 = lm(@formula(retorno ~ liquidez), df_empresas[!, 2:6])
r2(lm2.model)
# Regressão Linear - Retorno ~ Disclosure
lm3 = lm(@formula(retorno ~ disclosure), df_empresas[!, 2:6])
r2(lm3.model)
# Regressão Linear - Retorno ~ Ativos + Liquidez + Disclosure
lm4 = lm(@formula(retorno ~ ativos + liquidez + disclosure), df_empresas[!, 2:6])
r2(lm4.model)


# Testando o modelo lm4
df_validation = df_empresas[!, 2:6]
df_validation.y = predict(lm4, df_validation)
df_validation


@df df_validation plot(df_validation.retorno, label = "actual")
@df df_validation plot!(df_validation.y, label = "predicted")
