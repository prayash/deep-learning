## Introduction

## 1 - Keras Overview

## 2 - Setting Up

## 3 - Creating a Neural Network in Keras

### The train-test-evaluation flow

The computer learns how to perform a function by looking at labeled training data. We use the model train-test-evaluation flow to achieve that.

We train the NN by showing it training data and the expected output for that data, and it has to figure out how to replicate the expected result. After training, we load up new data (test data) to make sure the model actually learned how to solve the problem. Once it's trained and tested, we use it in the real world. This is the evaluation phase.

Keras makes it easy to set up a train, test, evaluation flow. First, we create our NN model.

```python
model = keras.models.Sequential()
```

Now, we can add layers to it by calling `model.add`:

```python
model.add(keras.layers.Dense())
```

The final step is to compile the model, that's when Keras actually builds a tensor flow model for us behind the scenes. We need to tell it how we want to measure the accuracy of each prediction made by the model during the training phase. We can choose several standard loss functions, or define our own. We need to tell it what optimizer algorithm as well.

```python
model.compile(loss='mean_squared_error', optimizer='adam')
```

To train the model, we call `model.fit` and pass in training data and expected output for the training data. Keras will run the training process and print out the progress to the console:

```python
model.fit(training_data, expected_output)
```

Once the training completes, it will report the final accuracy that was achieved with the training data. Once the model is trained, we're ready for the testing phase.

We can test the model by calling `model.evaluate` and passing in the testing data set and the expected output:

```python
error_rate = model.evaluate(testing_data, expected_output)
```

Once we are happy with the accuracy of the system, we can save the training model to a file. To do that call `model.save`. This file will contain everything we need to use our model in another program.

```python
model.save("trained_model.h5")
```

Now that we have a trained model, we're ready for the evaluation phase. We can load our previously trained model by calling `keras.models.load_model` function, and then use the model to make new predictions:

```python
model = keras.models.load_model('trained_model.h5')
predictions = model.predict(new_data)
```

### Keras Sequential API

A neural network (NN) is a machine-learning algorithm mad eup of individual nodes called neurons. These nodes, or neurons, are arranged into a serious of groups called layers. Nodes in each layer are connected to nodes in the following layer. Data flows from the input to the output along these connections. Each individual node is trained to perform a simple mathematical calculation and then feed its data to all the nodes it's connected to.

![Neural Network](img/nn.png)

When designing a NN in Keras, we have to decide 3 things:
- How many layers there should be
- How many nodes should be in each layer
- How the layers should be connected to each other

Bigger models with more layers and nodes can model more complex systems, but if you make it too big, it will beslow the train and is likely to overfit the data set (which is bad).

To build a NN in Keras, we use the sequential model API. It's called the sequential model API because we first need to create an empty model object, and then add layers to it one after another in sequence. Example:

```python
# Empty NN
model = keras.models.Sequential()

# New densely connected layer of 32 nodes
# Every node is connected to every node in the previous layer
# Since this is the very first layer, we also have to tell it
# how many input nodes there are
model.add(Dense(32, input_dim=9))
```

We can continue adding layers the same way:

```python
# Adds another layer with 128 densely connected nodes
model.add(Dense(128))

# Adds the final layer with one output node
model.add(Dense(1))
```

Keras is designed to make it quick to code the NN, but it still tries to give you a large amount of control over the structure of each layer.

Before values flow from nodes in one layer to the next, they pass through an activation function. Keras lets us choose which activation function is used for each layer. In this case, we've used a rectified linear unit, or RELU, activation function.

```python
model.add(Dense(num_of_neurons, activation='relu'))
```

Keras supports all standard activation functions in use today, and includes even esoteric ones that aren't widely used outside of research. 

There's also lots of less commonly needed things that we can customize in each layer, but most of the time, just choosing the number of nodes and layer and an activation function is good enough.

Keras also supports many different types of NN layers such as convolutional layers. These are typically used to process images or spatial data.

```python
keras.layers.convolutional.Conv2D()
keras.layers.recurrent.LSTM()
```

Recurrent layers are special layers that have a memory built into each neuron. These are used to process sequential data like words in a sentence where the previous data points are import to understanding the next data point. You can mix layers of different types in the same model as needed.

The final step of defining a model is to compile it by calling `model.compile`. We can pass in the optimizer algorithm and the loss function we want to use. 

The optimizer algorithm is the algorithm used to train your neural network. The loss function is how the training process measures how 'right' or 'wrong' the NN's predictions are.

### Pre-processing training data
Navigate to the `03` directory. We'll use `sales_data_training.csv` to train a NN that will predict how much money we can expect future video games to earn based on historical data. In the dataset, we have one row for eac video game title that the store has sold in the past. For each game, we have recorded several attributes:

![Training Data](img/training_data.png)

We'll use Keras to train the NN that tries to **predict the total earnings of a new game based on these characteristics**.

Notice how the values are between a range 0 all the way up to large numbers in the `total_earnings` column. To use this data to train a NN, we first have to scale the data so each value is between 0 and 1. NNs train best when data in each column is all scaled to the same range.

Along with `sales_data_training.csv` file, we also have a `sales_data_test.csv` file. The ML system will only be exposed to the training dataset, then we'll use the test dataset to check how well our NN is performing.

Open up `preprocess_data.py`. Let's code:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv")

# Data needs to be scaled to a small range like 0 to 1 for the NN to work well.
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
scaled_training = scaler.fit_transform(training_data_df)

# Calling transform instead of fit_transform applies the same amount of scaling as training data
scaled_testing = scaler.transform(test_data_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("sales_data_test_scaled.csv", index=False)
```

Once this runs, it will print to the console how much the data was scaled by and will also save our pre-processed CSV files in the same folder.

### Define a Keras model using the Sequential API

