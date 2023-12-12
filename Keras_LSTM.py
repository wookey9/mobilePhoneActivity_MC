# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator

def create_sequences_with_stride(data, input_sequence_length, output_sequence_length, stride):
    X, y = [], []
    for i in range(0, len(data) - input_sequence_length - output_sequence_length + 1, stride):
        X.append(data[i:(i + input_sequence_length)])
        y.append(data[(i + input_sequence_length):(i + input_sequence_length + output_sequence_length)])
    return np.array(X), np.array(y)


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
x_train, y_train = create_sequences_with_stride(train_data, input_sequence_length, output_sequence_length, 12)
x_test, y_test = create_sequences_with_stride(test_data, input_sequence_length, output_sequence_length, 12)

# Build the LSTM model
model = Sequential([
    LSTM(48, activation='relu', input_shape=(input_sequence_length, 1)),
    Dense(output_sequence_length)
])
# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, y_train,batch_size=10,validation_split=0.1, epochs=100)

# Evaluate the model on the test data
loss = model.evaluate(x_test, y_test)
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