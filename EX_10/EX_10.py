import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df= pd.read_csv('heart.csv')
X=np.array(df[['age','sex','exng','caa','cp','trtbps','chol','fbs','restecg','thalachh']])
scaler=preprocessing.StandardScaler()
Xscaled=scaler.fit_transform(X)
enc=preprocessing.LabelEncoder()
'''for i in range(1,25):
    input_vars.append('Sensor'+str(i))'''


#Y=np.array(df[['output']])
Y= np.array(enc.fit_transform(df['output']))

Xtrain, Xtest, Ytrain ,Ytest=train_test_split(Xscaled,Y,test_size=0.25)

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