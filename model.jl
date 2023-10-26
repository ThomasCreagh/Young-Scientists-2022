using Flux, MLDatasets
using Flux: train!, onehotbatch
using ProgressBars
using BSON: @save, @load
using DelimitedFiles

using CSV, DataFrames

data = readdlm("james.csv", ',', Int)
@load "300_bin.bson" model

println(findmax(model(Flux.flatten(data)[1, :]))[2] - 1)


# csv_file = CSV.read("train_bin.csv", DataFrame; header=2)
# data = Matrix(csv_file)'



# data_dev = data[:, 1:1001] # 1000 lines of file
# y_test = data_dev[1, :] # the value of the rest of the strings of numbers
# x_test = data_dev[2:end, :] # the string of numbers correlataing to each pixel

# data_train = data[:,1001:end] # all the lines from line 1001 to the end 
# y_train = data_train[1, :] # the value of the rest of the strings of numbers
# x_train = data_train[2:end, :] # the string of numbers correlataing to each pixel


# x_train = Float32.(x_train)
# y_train = Flux.onehotbatch(y_train, 0:9)

# println(size(y_train))
# println(size(x_train))

# function build(epochs, model_name)
#     model = Chain(
#         Dense(784, 1000, relu),
#         Dense(1000, 1000, relu),
#         Dense(1000, 10, sigmoid), softmax
#     )

#     loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
#     optimizer = ADAM(0.0001)
#     parameters = Flux.params(model)
#     train_data = [(Flux.flatten(x_train), y_train)]
#     test_data = [(Flux.flatten(x_test), y_test)]

#     for _ in ProgressBar(1:epochs)
#         println("before train")
#         # println(size(loss))
#         # println(size(parameters))
#         println(size(train_data))
#         # println(size(optimizer))
#         Flux.train!(loss, parameters, train_data, optimizer)
#         println("after train")
#     end

#     accuracy = 0
#     for i in 1:length(y_test)
#         if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
#             accuracy = accuracy + 1
#         end
#     end

#     @save model_name model

#     seven = false
#     if findmax(model(test_data[1][1][:, 1]))[2] - 1  == y_test[1]
#         seven = true
#     end

#     return accuracy / length(y_test), seven
# end

# build(300, "300_bin.bson")

# # 300 epochs is great! 94 accuracy