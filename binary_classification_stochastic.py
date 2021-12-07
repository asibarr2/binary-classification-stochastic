# import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

# Assume that the data files are in the following folder -- THIS WILL BE USED BY THE TA
drive.mount('/content/drive')
basePath = "/content/drive/My Drive/Assignment3/"

# Data file name variables
train = basePath + "gd-train.dat"
test = basePath + "gd-test.dat"

# Read the training and testing data files
df = pd.read_csv(train, sep = '\t')
dt = pd.read_csv(test, sep = '\t')

print(df[:].values)
print(dt[:].values)

# Activation Function - implement Sigmoid
def activation_function(h):
    # given 'h' compute and return 'z' based on the activation function implemented
    z  = 1 / (1 + np.exp(-h))
    return z

#For Debugging Purposes
print(activation_function(0.5))

# Train the model using the given training dataset and the learning rate
# return the "weights" learnt for the perceptron - include the weight assocaited with bias as the last entry
def train_weights(train_data, learning_rate=0.005):
    # initialize weights to 0
    # go through each training data instance
        # get 'x' as one multi-variate data instance and 'y' as the ground truth class label
        # obtain h(x)
        # call the activation function with 'h' as parameter to obtain 'z'
        # update all weights individually using learning_rate, (y-z), and the corresponding 'x'
    # return the final learnt weights

    weights = np.zeros([13,1], dtype=int)
    for index, row in df.iterrows():
      #Iterate through all row values of "A1" - "A13"
      x = row.values[:-1]
      # Iterate through all row values of "C"
      y = row.values[13]
      h = x.dot(weights)
      z = activation_function(h)
      error = z - y
      weights = weights + (learning_rate * x * error)
    return weights

# For Debugging Purposes
print(train_weights(df))
weights = train_weights(df)

# Test the model (weights learnt) using the given test dataset
# return the accuracy value
def test(test_data, weights, threshold):
    # go through each testing data instance
        # get 'x' as one multi-variate data instance and 'y' as the ground truth class label
        # obtain h(x)
        # call the activation function with 'h' as parameter to obtain 'z'
        # use 'threshold' to convert 'z' to either 0 or 1 so as to match to the ground truth binary labels
        # compare the normalized 'z' with 'y' to calculate the positive and negative instances for calculating accuracy
    # return the accuracy value for the given test dataset
    count = 0
    for index, row in dt.iterrows():
      x = row.values[:-1]
      h = x.dot(weights)
      z  = activation_function(h)
      if z.size > threshold:
        z = 1
        count += 1
      else:
        z = 0

    accuracy = 100 * count / len(test_data)
    return accuracy

# For Debugging Purposes

print(test(dt, weights, 0.05))

# Gradient Descent function
def gradient_descent(df_train, df_test, learning_rate=0.05, threshold=0.5):
    # call the train function to train the model and obtain the weights
    # call the test function with the training dataset to obtain the training accuracy
    # call the test function with the testing dataset to obtain the testing accuracy
    # return (trainAccuracy, testAccuracy)
    weight = train_weights(df_train)
    trainAccuracy = test(df_train, weight, threshold)
    testAccuracy = test(df_test, weight, threshold)
    return (trainAccuracy, testAccuracy)

print(gradient_descent(df, dt))

# Threshold of 0.5 will be used to classify the instance for the test. If the value is >= 0.5, classify as 1 or else 0.
threshold = 0.5

# Main algorithm loop
# Loop through all the different learning rates [0.05, 1]
    # For each learning rate selected, call the gradient descent function to obtain the train and test accuracy values
    # Print both the accuracy values as "Accuracy for LR of 0.1 on Training set = x %" OR "Accuracy for LR of 0.1 on Testing set = x %"
for i in np.arange(0.05, 1.0, 0.01):
  accuracy = gradient_descent(df,dt, i, threshold)
  print("Accuracy for LR of {} on Training set = {}".format(i, accuracy[0]))
  print("Accuracy for LR of {} on Testing set = {}".format(i, accuracy[1]))

# Plot the graphs for accuracy results.
# There will be 2 graphs - one for training data and the other for testing data
# For each graph,
    # X-axis will be the learning rate going from 0.05-1 in increments on 0.05
    # Y-axis will be the accuracy values at the selected learning rate.

plt.plot(accuracy[0])
plt.show()
