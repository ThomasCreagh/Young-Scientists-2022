using CSV, DataFrames

csv_file = CSV.read("train.csv", DataFrame; header=2)
n, m = size(data)
data = Matrix(csv_file)'



data_dev = data[:,1:1001]
y_data = data_dev[1, :]
x_data = data_dev[2:end, 1]

data_train = data[:,1001:end]
y_train = data_train[1, :]
x_train = data_train[2:end, 1]


function init_params()
    w1 = rand(10, 784) .-.5
    b1 = rand(10, 1) .-.5
    w2 = rand(10, 10) .-.5
    b2 = rand(10, 1) .-.5

    return w1, b1, w2, b2
end


relu(z) = max(0, z)
reLU_deriv(z) = z > 0
softmax(z) = exp(z) / sum(exp(z))
one_hot(y) = unique(y)' .== permutedims(y)'

function forward_prop(w1, b1, w2, b2, x)
    z1 = sum(w1 .* x) + b1
    a1 = relu(z1)
    z2 = sum(w2 .* a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2
end


function backward_prop(z1, a1, z2, a2, w2, y)
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * sum(dz2 .* a1')
    db2 = 1 / m * sum(dz2)
    dz1 = w2
