import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')


data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255.

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.
_,m_train = x_train.shape




class MNISTalgorithm:
    def __init__(self, epochs=500, alpha=0.1):
        self.epochs = epochs
        self.alpha = alpha
    
    
    def initialization(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2
    

    def relu(self, z):
        return np.maximum(z, 0)
    
    def relu_deriv(self, z):
        return z > 0

    def softmax(self, z):
        return np.exp(z) / sum(np.exp(z))
    

    def forward(self, w1, b1, w2, b2, x):
        z1 = w1.dot(x) + b1
        a1 = self.relu(z1)
        z2 = w2.dot(a1) + b2
        a2 = self.softmax(z2)
        return z1, a1, z2, a2
    

    def backward(self, z1, a1, z2, a2, w1, w2, x, y):
        one_hot_y = self.one_hot(y)
        dz2 = a2 - one_hot_y
        dw2 = 1 / m * dz2.dot(a1.T)
        db2 = 1 / m * np.sum(dz2)
        dz1 = w2.T.dot(dz2) * self.relu_deriv(z1)
        dw1 = 1 / m * dz1.dot(x.T)
        db1 = 1 / m * np.sum(dz1)
        return dw1, db1, dw2, db2


    def one_hot(self, y):
        one_hot_y = np.zeros((y.size, y.max() + 1))
        one_hot_y[np.arange(y.size), y] = 1
        one_hot_y = one_hot_y.T
        return one_hot_y
    

    def update(self, w1, b1, w2, b2, dw1, db1, dw2, db2):
        w1 = w1 - self.alpha * dw1
        b1 = b1 - self.alpha * db1
        w2 = w2 - self.alpha * dw2
        b2 = b2 - self.alpha * db2
        return w1, b1, w2, b2


    def predictions(self, a2):
        return np.argmax(a2, 0)


    def gradients(self, x, y):
        w1, b1, w2, b2 = self.initialization()
        for i in range(self.epochs):
            z1, a1, z2, a2 = self.forward(w1, b1, w2, b2, x)
            dw1, db1, dw2, db2 = self.backward(z1, a1, z2, a2, w1, w2, x, y)
            w1, b1, w2, b2 = self.update(w1, b1, w2, b2, dw1, db1, dw2, db2)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.predictions(a2)
                print(self.accuracy(predictions, y))
        return w1, b1, w2, b2


    def make_predictions(self, x, w1, b1, w2, b2):
        _, _, _, a2 = self.forward(w1, b1, w2, b2, x)
        predictions = self.predictions(a2)
        return predictions
    

    def accuracy(self, predictions, y):
        print(predictions, y)
        return np.sum(predictions == y) / y.size
    

    def test_prediction(self, index, w1, b1, w2, b2):
        current_image = x_train[:, index, None]
        prediction = self.make_predictions(x_train[:, index, None], w1, b1, w2, b2)
        label = y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()


model = MNISTalgorithm(500, 0.1)

w1, b1, w2, b2 = model.gradients(x_train, y_train)

model.test_prediction(0, w1, b1, w2, b2)
model.test_prediction(1, w1, b1, w2, b2)
model.test_prediction(2, w1, b1, w2, b2)
model.test_prediction(3, w1, b1, w2, b2)

dev_predictions = model.make_predictions(x_dev, w1, b1, w2, b2)
print(model.accuracy(dev_predictions, y_dev))