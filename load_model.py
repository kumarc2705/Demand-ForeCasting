from keras.models import load_model
import numpy

model = load_model('Hackathon-NN.h5')
xnew=numpy.array([11796,128,18964,13208,0.0108511359782977,271,217 ,-54])#Input parameters for current week as a 1D array
'''
Put i/p in the format
as mentioned below
numpy.array([Total Views , Number of Products sold , Revenue ,OOS Instances , Order Conversion ratio ,
Demand for previous week ,Demand For previous to previous week ,(difference in both the adjacent demands i.e (previous week -previous to previous week)])
Example is shown above for 41st column in the dataset mentioned in xls
'''
xnew=numpy.array([xnew])
ynew=model.predict(xnew)#output / prediction of demand for next week
print(ynew)
