import numpy as np
import math

# Multilayer Perceptron Class
class MultiLayerPerceptronBinary:
    # Initialization Function
    def __init__(self, alpha, steps) -> None:
        self.alpha = alpha
        self.steps = steps
        self.input_weights = np.empty([50,784])
        self.hidden_weights = None
        self.input_bias = 0
        self.hidden_bias = 0

    # Fit method
    def fit(self, data, labels, accuracy):
        # Randomize input layer weights and bias
        self.input_bias = np.random.uniform(-1,1,50)
        for i in range(50):
            self.input_weights[i] = (np.random.uniform(-1,1,len(data[0])))
        
        # Randomize hidden layer weights and bias
        self.hidden_bias = np.random.uniform(-1,1)
        self.hidden_weights = np.random.uniform(-1,1,50)

        # Train for set time steps
        for i in range(self.steps):
            # For every example in the data
            for j in range(len(data)):
                # Compute forward pass
                hiddenLayer_in = np.dot(data[j], self.input_weights.transpose()) + self.input_bias
                hiddenLayer_out = np.array(list(map(self.activate, hiddenLayer_in)))

                prediction = self.activate(np.dot(hiddenLayer_out,self.hidden_weights) + self.hidden_bias)
                
                # Backpropagation
                delta_out = (labels[j]-prediction)*(prediction)*(1-prediction)
                delta_hidden = hiddenLayer_out*(-hiddenLayer_out+1)*(self.hidden_weights)*delta_out

                # Update weight values
                self.hidden_weights += self.alpha*hiddenLayer_out*delta_out
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
            prediction = self.activate(np.dot(hiddenLayer_out,self.hidden_weights) + self.hidden_bias)

            if prediction <= .5:
                prediction = 0
            else: prediction = 1

            if prediction == labels[i]: correct += 1
        
        return 100*(correct/len(data))

    # Sigmoid activation function
    def activate(self, x):
        return 1/(1+math.pow(math.e,-x))

                

