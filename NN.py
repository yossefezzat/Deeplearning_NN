from data_processing import final_data , WIDTH , HEIGHT
import tensorflow as tf
from tensorflow import keras
import numpy as np

image_size = WIDTH*HEIGHT


def extract_train_valid_test():
    train_data , validation_data ,test_data = final_data()

    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test =  list()
    X_validation = list()
    Y_validation = list()
    
    for i in range(len(train_data)):
        X_train.append(train_data[i][:900])
        Y_train.append(train_data[i][900:])
    
    for i in range(len(validation_data)):
        X_validation.append(train_data[i][:900])
        Y_validation.append(train_data[i][900:])
    
    for i in range(len(test_data)):
        X_test.append(train_data[i][:900])
        Y_test.append(train_data[i][900:])
    return X_train, Y_train , X_validation , Y_validation , X_test , Y_test


def k_flud_arch1_NN(num):
    accuracies = []
    for i in range(num):
        X_train, Y_train , X_validation , Y_validation , X_test , Y_test = extract_train_valid_test()
        model = keras.Sequential()
        model.add(keras.layers.Dense(220, activation='relu'))
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, Y_train ,epochs=20 , batch_size=64)
        
        test_loss, test_acc = model.evaluate(X_test , Y_test)
        print(test_loss , test_acc)
        
        predictions = model.predict(X_test)
        print(Y_test[0])
        print(np.argmax(predictions[0]), predictions[0])   
        accuracies.append(test_acc)
    return accuracies

accuracies = k_flud_arch1_NN(10)

print(accuracies)