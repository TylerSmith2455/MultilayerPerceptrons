import pandas as pd
import numpy as np
from multilayerPerceptronBinary import MultiLayerPerceptronBinary
from multiLayerPerceptronExtra import MultiLayerPerceptronExtra

def main():
    # Read in training data
    trainData = pd.read_csv('mnist_train_0_1.csv', header=None).to_numpy()
    trainLabels = trainData[:,0]
    trainData = (1/255) * np.delete(trainData, 0, 1)
    
    multiLayerPerceptron = MultiLayerPerceptronBinary(.5,100)
    multiLayerPerceptron.fit(trainData,trainLabels,99)

    # Read in test data
    testData = pd.read_csv('mnist_test_0_1.csv', header=None).to_numpy()
    testLabels = testData[:,0]
    testData = (1/255) * np.delete(testData, 0, 1)

    print("\n","             Main Assignment")
    print("mnist_test_0_1 accuracy:", multiLayerPerceptron.predict(testData,testLabels),"%")

    # Read in training data
    trainData = pd.read_csv('mnist_train_0_4.csv', header=None).to_numpy()
    trainLabels = trainData[:,0]
    trainData = (1/255) * np.delete(trainData, 0, 1)
    
    multiLayerPerceptron = MultiLayerPerceptronExtra(.5,100)
    multiLayerPerceptron.fit(trainData,trainLabels,95)

    # Read in test data
    testData = pd.read_csv('mnist_test_0_4.csv', header=None).to_numpy()
    testLabels = testData[:,0]
    testData = (1/255) * np.delete(testData, 0, 1)
    
    print("\n","             Extra Credit")
    print("mnist_test_0_4 accuracy:", multiLayerPerceptron.predict(testData,testLabels),"%")

if __name__ == "__main__":
    main()