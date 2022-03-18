import numpy as np
import math

# Multilayer Perceptron Class
class MultiLayerPerceptronExtra:
    # Initialization Function
    def __init__(self, alpha, steps) -> None:
        self.alpha = alpha
        self.steps = steps
        self.input_weights = np.empty([50,784])
        self.hidden_weights = np.empty([5,50])
        self.input_bias = 0
        self.hidden_bias = 0
        self.classes = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])

    # Fit method
    def fit(self, data, labels, accuracy):
        # Randomize input layer weights and bias
        self.input_bias = np.random.uniform(-1,1,50)
        for i in range(50):
            self.input_weights[i] = np.random.uniform(-1,1,len(data[0]))
        
        # Randomize hidden layer weights and bias
        self.hidden_bias = np.random.uniform(-1,1,5)
        for i in range(5):
            self.hidden_weights[i] = np.random.uniform(-1,1,50)

        # Train for set time steps
        for i in range(self.steps):
            # For every example in the data
            for j in range(len(data)):
                # Compute forward pass
                hiddenLayer_in = np.dot(data[j], self.input_weights.transpose()) + self.input_bias
                hiddenLayer_out = np.array(list(map(self.activate, hiddenLayer_in)))

                prediction_in = np.dot(hiddenLayer_out,self.hidden_weights.transpose()) + self.hidden_bias
                prediction_out = np.array(list(map(self.activate, prediction_in)))

                # Backpropagation
                delta_out = (self.classes[labels[j]]-prediction_out)*(prediction_out)*(1-prediction_out)
                delta_hidden = hiddenLayer_out*(-hiddenLayer_out+1)*np.dot(self.hidden_weights.transpose(),delta_out)

                # Update weight values
                self.hidden_weights += self.alpha*delta_out.reshape(5,1)*hiddenLayer_out
                self.input_weights += self.alpha*delta_hidden.reshape(50,1)*data[j].reshape(1,784)

                self.hidden_bias += self.alpha*delta_out
                self.input_bias += self.alpha*delta_hidden
            
            if self.predict(data, labels) > accuracy:
                return
            
    # Use the neural network model to predict values
    def predict(self, data, labels):
        correct = 0
        
        for i in range(len(data)):
            hiddenLayer_in = np.dot(data[i], self.input_weights.transpose()) + self.input_bias
            hiddenLayer_out = np.array(list(map(self.activate, hiddenLayer_in)))

            prediction_in = np.dot(hiddenLayer_out,self.hidden_weights.transpose()) + self.hidden_bias
            prediction_out = np.array(list(map(self.activate, prediction_in)))

            for j in range(5):
                if prediction_out[j] <= .5:
                    prediction_out[j] = 0
                else: prediction_out[j] = 1
            
            if np.array_equal(prediction_out, self.classes[labels[i]]): correct += 1
        
        return 100*(correct/len(data))

    # Sigmoid activation function
    def activate(self, x):
        return 1/(1+math.pow(math.e,-x))