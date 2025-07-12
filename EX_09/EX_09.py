import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import tensorflow as tf 

df= pd.read_csv('sensor_readings_24.csv')

input_vars=[]
for i in range(1,25):
    input_vars.append('Sensor'+str(i))

X= np.array(df[input_vars])
scaler=preprocessing.StandardScaler()
Xscaled=scaler.fit_transform(X)
#enc=preprocessing.LabelEncoder()
Ytemp= pd.get_dummies(df['Command'])
Y=np.array(Ytemp)

Xtrain, Xtest, Ytrain ,Ytest=train_test_split(Xscaled,Y,test_size=0.25)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation=tf.nn.relu,input_shape=(Xtrain.shape[1],)),
    tf.keras.layers.Dense(50,activation=tf.nn.relu),
    tf.keras.layers.Dense(30,activation=tf.nn.relu),
   tf.keras.layers.Dense(Ytrain.shape[1],activation=tf.nn.softmax)
   ])


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])



model.fit(Xtrain,Ytrain,validation_data=(Xtest,Ytest),epochs=50,batch_size=100)

#%%

trainpredictions= np.argmax(model.predict(Xtrain),axis=1)
testpredictions= np.argmax(model.predict(Xtest),axis=1)

#%%

print('Accuracy in tain data %.2f'% accuracy_score(np.argmax(Ytrain,axis=1),trainpredictions))
print('Accuracy in test data %.2f'% accuracy_score(np.argmax(Ytest,axis=1),testpredictions))

#%%

mapping={0:Ytemp.columns[0],1:Ytemp.columns[1],
         2:Ytemp.columns[2],3:Ytemp.columns[3]}

testpredictionstrings=[]
for number in testpredictions:
    testpredictionstrings.append(mapping[number])
    
    realcommands=[]
    for number in np.argmax(Ytest,axis=1):realcommands.append(mapping[number])

dfvalidation= pd.DataFrame()
dfvalidation['Prediction']=testpredictionstrings
dfvalidation['Real Command']=realcommands
dfsample= dfvalidation.sample(20)