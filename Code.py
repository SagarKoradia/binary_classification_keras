from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)

fn = r'C:\Users\DELL I5558\Desktop\Python\electricity_price_and_demand_20170926.csv'
dataset = np.loadtxt(fn, delimiter=",")
X = dataset[:, 0:5]
Y = dataset[:, 5]

model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(8, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
