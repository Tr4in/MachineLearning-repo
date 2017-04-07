from  keras.models import Sequential
from keras.layers import Dense
import numpy

# Random seed
numpy.random.seed(10)

# open the file first and extract the dataset
dataset = numpy.loadtxt("abalone.csv", usecols=(1,8),delimiter=",")

# Train input X and output Y
X = dataset[:,1:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile data
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epocs=150, batch_size=10)

scores = model.evaluate(X,Y)
print model.metrics_names[1], scores[1] * 100
