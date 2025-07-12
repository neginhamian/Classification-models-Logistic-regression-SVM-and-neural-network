# Excercise 7 

A SCITOS G5 robot was remote controlled along a wall while the readings of its 24 ultrasonic sensors were recorded. By using this data, build an AI model which can independently drive the robot along a wall. Separate 25 % of the data into a test dataset to validate the accuracy of your model. Use the logistic regression model as your machine learning algorithm.

# Exercise 8 

Repeat the previous exercise but use the Support Vector Machine as your machine learning algorithm. Report the accuracy of your model in test data, and also print the predicted and the actual control command for 20 random rows from the test data.

# Exercise 9 

Repeat the previous exercise but use the Neural Network as your machine learning algorithm. Report the accuracy of your model in test data, and also print the predicted and the actual control command for 20 random rows from the test data.

# Exercise 10

Use the dataset heart.csv to build neural network, which predicts if the patient has an increased chance of heart attack (variable Output in heart.csv). Separate 25 % of the data into a test data. Report the accuracy of your model in both train and test datasets.

Explanation of the variables (treat “cp” as a categorical variable):

* Age : Age of the patient

* Sex : Sex of the patient

* exang: exercise induced angina (1 = yes; 0 = no)

* ca: number of major vessels (0-3)

* cp : Chest Pain type chest pain type

Value 1: typical angina

Value 2: atypical angina

Value 3: non-anginal pain

Value 4: asymptomatic

* trtbps : resting blood pressure (in mm Hg)

* chol : cholestoral in mg/dl fetched via BMI sensor

* fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

* rest_ecg : resting electrocardiographic results

Value 0: normal

Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

* thalach : maximum heart rate achieved

* output : 0= less chance of heart attack 1= more chance of heart attack
