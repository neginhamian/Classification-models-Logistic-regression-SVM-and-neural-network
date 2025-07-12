import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df= pd.read_csv('sensor_readings_24.csv')

input_vars=[]
for i in range(1,25):
    input_vars.append('Sensor'+str(i))

X= np.array(df[input_vars])
enc=preprocessing.LabelEncoder()
Y= np.array(enc.fit_transform(df['Command']))

Xtrain, Xtest, Ytrain ,Ytest=train_test_split(X,Y,test_size=0.25)

model= SVC()

model.fit(Xtrain,Ytrain)
trainpredictions= model.predict(Xtrain)
testpredictions= model.predict(Xtest)


print('Accuracy in tain data %.2f'% accuracy_score(Ytrain,trainpredictions))
print('Accuracy in test data %.2f'% accuracy_score(Ytest,testpredictions))

#%%
testpredictionstrings= enc.inverse_transform(testpredictions)
dfvalidation= pd.DataFrame()
dfvalidation['Prediction']=testpredictionstrings
dfvalidation['Real Command']=enc.inverse_transform(Ytest)
dfsample= dfvalidation.sample(20)