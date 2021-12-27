using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics

# REGRESSÃO NÃO LINEAR MÚLTIPLA
# Carregamento da base de dados

function dados_empresas()
    path = "empresas.csv"
    df = CSV.read(path, DataFrame)
    select(df, Not([:Column1]))
end

df = dados_empresas()

# Primeiros 5 registros dos dados
first(df, 5)

# Descrição do dataset
describe(df)

# Visualização dos dados - Correlação
@df df corrplot(cols(2:6))

cor(Matrix(df[!, 2:6]))