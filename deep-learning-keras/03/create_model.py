import pandas as pd
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

# Drop the total earnings column
# X now contains all the input features for each game
X = training_data_df.drop('total_earnings', axis=1).values

# Grab just the total earnings column
# Y now contains only the expected earnings for each game
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()

# 9 characteristics for each dataset, so 9 input dimensions!
# Rectified Linear Unit Activation Function will let us model more complex
# and non-linear functions
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")