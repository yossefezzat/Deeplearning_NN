from data_processing_CNN import final_data , WIDTH , HEIGHT
from tensorflow import keras
import numpy as np

image_size = WIDTH*HEIGHT*3


def extract_train_valid_test():
    train_data , validation_data , test_data = final_data()
    
    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test =  list()
    X_validation = list()
    Y_validation = list()
        
    for i in range(len(train_data)):
        X_train.append(train_data[i][0])
        Y_train.append(train_data[i][1])
    Y_train = [list(map(int, i)) for i in Y_train]
    for i in range(len(validation_data)):
        X_validation.append(validation_data[i][0])
        Y_validation.append(validation_data[i][1])
    Y_validation = [list(map(int, i)) for i in Y_validation]    
    for i in range(len(test_data)):
        X_test.append(test_data[i][0])
        Y_test.append(test_data[i][1])
    Y_test = [list(map(int, i)) for i in Y_test]  

    return X_train, Y_train , X_validation , Y_validation , X_test , Y_test

def build_model(n_layers=[], activ_func_layers=[]):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, 
                                  kernel_size = (3, 3),
                                  activation = 'relu',
                                  input_shape = (30 , 30 , 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


def k_fold_arch_CNN(n_layers, activ_func_layers, k_fold=5):
    history_acc = []
    model = build_model(n_layers, activ_func_layers)
    for i in range(k_fold):
        X_train, Y_train , X_validation , Y_validation , X_test , Y_test = extract_train_valid_test()
        history = model.fit(np.array(X_train), np.array(Y_train) ,epochs=10 , batch_size=64)        
        test_loss, test_acc = model.evaluate(np.array(X_train) , np.array(Y_train))        
        Y_pred = model.predict(np.array(X_test))
        Y_pred = [int(np.argmax(i)) for i in Y_pred]
        history_acc.append(history.history['accuracy'][-1]) #last epoch acc
    return history_acc

def experiment(n_layers, activ_func_layers, k_fold=3):
    history_acc = k_fold_arch_CNN(n_layers, activ_func_layers, k_fold)
    print(history_acc)
    print('model ACC: ', round(sum(history_acc)/len(history_acc)*100, 1), '%')
'''
experiment([450, 220, 110, 50, 20, 10],
           ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'])
'''
'''
experiment([220, 50, 10],
           [relu', 'relu', softmax'])
'''
'''
experiment([180, 30, 10],
           ['relu', 'relu', 'softmax'])
'''

experiment([100, 10],
           ['relu', 'softmax'])





