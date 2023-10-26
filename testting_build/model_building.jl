using Flux, MLDatasets, CUDA
using Flux: train!, onehotbatch
using ProgressBars
using BSON: @save, @load
using CSV


x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
x_train = Float32.(x_train)
y_train = Flux.onehotbatch(y_train, 0:9)

println(size(y_train))
println(size(x_train))


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

#     println(size(train_data))

#     for _ in ProgressBar(1:epochs)
#         Flux.train!(loss, parameters, train_data, optimizer)
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


# function test(model_name)
#     @load model_name model

#     test_data = [(Flux.flatten(x_test), y_test)]

#     accuracy = 0
#     for i in 1:length(y_test)
#         if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
#             accuracy = accuracy + 1
#         end
#     end

#     seven = false
#     if findmax(model(test_data[1][1][:, 1]))[2] - 1  == y_test[1]
#         seven = true
#     end

#     return accuracy / length(y_test), seven
# end


# println("build(1) or test(2) model?: ")
# input = readline()


# if input == "1"
#     println("how many epochs?: ")
#     epochs = parse(Int, readline())
#     println("what do you want to name the model?: ")
#     model_name = readline()
#     println(build(epochs, model_name))

# elseif input == "2"
#     println("what is the model name?: ")
#     model_name = readline()
#     println(test(model_name))

# else
#     println("ERROR: invalid input")

# end




# # 300 epochs is great! 94 accuracy