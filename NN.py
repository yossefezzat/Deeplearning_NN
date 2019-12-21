from data_processing import final_data , WIDTH , HEIGHT
import tensorflow as tf
from tensorflow import keras
import numpy as np



train_data , validation_data ,test_data = final_data()

image_size = WIDTH*HEIGHT
X_train = list()
Y_train = list()
for i in range(len(train_data)):
    X_train.append(train_data[i][:900])
    Y_train.append(train_data[i][900:])



X_validation = list()
Y_validation = list()

for i in range(len(validation_data)):
    X_validation.append(train_data[i][:900])
    Y_validation.append(train_data[i][900:])


X_test = list()
Y_test =  list()

for i in range(len(test_data)):
    X_test.append(train_data[i][:900])
    Y_test.append(train_data[i][900:])



model = keras.Sequential()


model.add(keras.layers.Dense(220, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train ,epochs=10 , batch_size=64)

#test_loss, test_acc = model.evaluate(X_train , Y_train)
#print(test_loss , test_acc)

predictions = model.predict(X_test)

    
    
print(predictions)
