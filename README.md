# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Design Steps

### Step 1:
Read and preprocess training data, including scaling and sequence creation.

### Step 2:
Initialize a Sequential model and add SimpleRNN and Dense layers.

### Step 3:
Compile the model with Adam optimizer and mean squared error loss.

### Step 4:
Train the model on the prepared training data.

### Step 5:
process test data, predict using the trained model, and visualize the results

## Program
#### Name: Aakash P
#### Register Number: 212222110001

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('/content/trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
model = Sequential()
model.add(layers.SimpleRNN(150,input_shape=(length,n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
print("NAME: Aakash P \nREGISTER NUMBER: 212222110001 \n        ")
model.summary()
test_set = dataset_test.iloc[:,1:2].values
dataset_test = pd.read_csv('/content/testset.csv')
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_train = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
    X_test.append(inputs_scaled[i-60:i,0])
    y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
print("Name: Aakash P , Register Number: 212222110001")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error as mse
print('Aakash P')
print(mse(y_test,predicted_stock_price))
```
## Output

### True Stock Price, Predicted Stock Price vs time
![5 1](https://github.com/user-attachments/assets/9d5016a6-e48e-4d26-97ac-ff2d37a361d1)

### Mean Square Error
![5 2](https://github.com/user-attachments/assets/99334beb-52b4-4ce1-adaa-7d9481bc34ea)

## Result
Hence the Recurrent Neural Network model for stock price prediction is developed
