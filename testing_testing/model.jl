using Flux
using BSON: @load
using DelimitedFiles
using MLDatasets

x_test, y_test = MNIST(split=:test)[:]

data = readdlm("james.csv", ',', Int)
@load "500_model" model

println(findmax(model(Flux.flatten(data)[1, :]))[2] - 1)
