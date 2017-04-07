from keras.models import Sequential
from keras.layers import Dense
import numpy

# Radom seed
numpy.random.seed(7)

# Load directory
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input(X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()

# Dense: 12 neurons in the first layers. Expecting 8 inputs
# Dense: 8 neurons in the second layer
model.add(Dense(12, input_shape=(13, None), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Minimize losses
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10d)

# Prediction Haze
predition_haze = mode.predict(X)

scores = model.evaluate(X, Y)
print(model.metrics_names[1], scores[1]*100)
