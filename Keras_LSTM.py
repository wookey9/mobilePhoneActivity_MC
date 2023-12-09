# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator

# Load the dataset
df = pd.read_csv('up_ratio.csv')

# Assuming the time-series values are in the second column
data = df.iloc[:, 1].values
data = data.reshape((-1, 1))

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Split data into training and testing sets
train_data, test_data = train_test_split(data_scaled, test_size=0.2, shuffle=False)

# Define sequence length for input and output
input_sequence_length = 336
output_sequence_length = 48

# Generator for the training data
train_generator = TimeseriesGenerator(train_data, train_data, length=input_sequence_length, batch_size=10)

# Generator for the testing data
test_generator = TimeseriesGenerator(test_data, test_data, length=input_sequence_length, batch_size=10)

# Build the LSTM model
model = Sequential()
model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(input_sequence_length, 1)))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu'))
model.add(Dense(output_sequence_length))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_generator, epochs=100)

# Evaluate the model on the test data
loss = model.evaluate(test_generator)
print(f'Test Loss: {loss}')

# Predict future values
last_sequence = train_data[-input_sequence_length:]
last_sequence = last_sequence.reshape((1, input_sequence_length, 1))
predicted_values = model.predict(last_sequence)

# Inverse transform to get actual values
predicted_values = scaler.inverse_transform(predicted_values)
predicted_values = predicted_values.flatten()

# Output the predicted future values
print(predicted_values)
model.save('lstm_model')