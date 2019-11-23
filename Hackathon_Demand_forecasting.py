import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense
'''
Steps to run the code : In th
'''
# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pd.read_excel(r'C:\Users\kumchand1\Desktop\ml proj\test_data\DemandForecasting.xlsx', sheet_name='Code_Format')   #put the path of existing data
dataset = dataframe.values
dataset = dataset.astype('float32')

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


def create_dataset(dataset , look_back): #dataset cleaning and formatting
        dataX=numpy.zeros(shape=(len(dataset)-look_back, len(dataset[0])+look_back))
        dataY=numpy.zeros((len(dataset) - look_back , 1))
        for i in range(len(dataset)-look_back):
            dataX[i][ : len(dataset[0])-1]= dataset[i+look_back][: len(dataset[0])-1]
            dataX[i][len(dataset[0])-1 : len(dataset[0])+look_back-1 ]=(dataset[i: i+look_up,len(dataset[0])-1]).transpose()
            dataX[i][len(dataset[0])+look_back-1]=dataset[i+look_up-1,len(dataset[0])-1]-dataset[i+look_up-1,len(dataset[0])-2]
            dataY[i][0]=dataset[i+look_back][len(dataset[0])-1]
        return dataX , dataY

# reshape into X=t and Y=t+1 
look_up=2
trainX, trainY = create_dataset(train , look_up)
testX, testY = create_dataset(test ,look_up)
print(trainX.shape)

# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(6+look_up,input_dim=6+look_up, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=2000, batch_size=2, verbose=2)

model.save('Hackathon-NN.h5')

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))





