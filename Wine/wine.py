from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(10)

dataset = np.loadtxt("wine.csv", delimiter=",")

X = dataset[:,0:13]
Y = dataset[:,0]

model = Sequential()
model.add(Dense(15, input_dim=13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='hinge', optimizer='adamax', metrics=['accuracy'])

model.fit(X, Y, epochs=300, batch_size=10)

#Prediction HazE
prediction = model.predict(X)

scores = model.evaluate(X,Y)
print (model.metrics_names[1], prediction[1] * 100)
