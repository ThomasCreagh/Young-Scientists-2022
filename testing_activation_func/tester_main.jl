using Flux, MLDatasets, CUDA
using Flux: train!, onehotbatch
using ProgressBars
using CSV


x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
x_train = Float32.(x_train)
y_train = Flux.onehotbatch(y_train, 0:9)


function nn(epochs, layer_1, layer_2, activation_0, activation_1, activation_2)
    model = Chain(
        Dense(784, layer_1, activation_0),
        Dense(layer_1, layer_2, activation_1),
        Dense(layer_2, 10, activation_2), softmax
    )

    loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
    optimizer = ADAM(0.0001)
    parameters = Flux.params(model)
    train_data = [(Flux.flatten(x_train), y_train)]
    test_data = [(Flux.flatten(x_test), y_test)]

    for _ in ProgressBar(1:epochs)
        Flux.train!(loss, parameters, train_data, optimizer)
    end

    accuracy = 0
    for i in 1:length(y_test)
        if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
            accuracy = accuracy + 1
        end
    end

    return accuracy / length(y_test)
end

function test_nn()
    nn_list = []
    epochs = 10
    layer_1 = 0
    layer_2 = 0
    activation_0 = relu
    activation_1 = relu
    activation_2 = relu
    count = 0

    for i in 1:3
        layer_1 = i*100

        for j in 1:3
            layer_2 = j*100

            for k in 1:2
                if k == 1
                    activation_0 = relu
                else
                    activation_0 = sigmoid
                end

                for l in 1:2
                    if l == 1
                        activation_1 = relu
                    else
                        activation_1 = sigmoid
                    end

                    for m in 1:2
                        if m == 1
                            activation_2 = relu
                        else
                            activation_2 = sigmoid
                        end
                        
                        accuracy = nn(epochs, layer_1, layer_2, activation_0, activation_1, activation_2)

                        push!(nn_list, [accuracy, epochs, layer_1, layer_2, activation_0, activation_1, activation_2])
                        
                        count = count + 1
                        total_iterations = 3 * 3 * 2 * 2 * 2
                        percentage_iterations = (count / total_iterations) * 100

                        println("$count of $total_iterations = $percentage_iterations%")
                        println("accuracy = $accuracy")

                    end
                end
            end
        end
    end
    
    return nn_list

end

list = test_nn()
println(list)
