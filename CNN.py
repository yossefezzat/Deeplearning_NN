from data_processing_CNN import final_data , WIDTH , HEIGHT
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

image_size = WIDTH*HEIGHT*3

models = []
histories = []

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
###############################################################################

def build_model(n_layers=[] , num_cov= 1 , activ_func_layers=[] , pooling=[]):
    print("poooooooooooooling" , pooling)
    model = keras.Sequential()
    for i in range(num_cov):
        model.add(keras.layers.Conv2D(32, 
                                      kernel_size = (2, 2),
                                      activation = 'relu',
                                      input_shape = (WIDTH , HEIGHT , 3)))
    model.add(keras.layers.MaxPooling2D(( pooling[0], pooling[1] )))
    model.add(keras.layers.Flatten())
    for i in range(len(n_layers)):
        model.add(keras.layers.Dense(n_layers[i], activation = activ_func_layers[i]))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model

################################################################################
    
def k_fold_arch_CNN(n_layers ,num_cov , activ_func_layers,pooling , k_fold):
    history_acc = []
    model = build_model(n_layers ,num_cov , activ_func_layers , pooling)
    models.append(model)
    for i in range(k_fold):
        X_train, Y_train , X_validation , Y_validation , X_test , Y_test = extract_train_valid_test()
        history = model.fit(np.array(X_train), np.array(Y_train) ,epochs=10  , validation_data = (np.array(X_validation) , np.array(Y_validation)) ,   batch_size=64)               
        test_loss, test_acc = model.evaluate(np.array(X_test) , np.array(Y_test))        
        Y_pred = model.predict(np.array(X_test))
        print(np.argmax(Y_pred[0]) , " hello  ", np.argmax(Y_test[0]))
        Y_pred = [int(np.argmax(i)) for i in Y_pred]
        history_acc.append(history.history['accuracy'][-1]) #last epoch acc
    return history_acc

accuraices = []
def experiment(n_layers, num_conv , activ_func_layers,pooling, k_fold=3):
    history_acc = k_fold_arch_CNN(n_layers , num_conv, activ_func_layers, pooling , k_fold)
    accuraices.append(round(sum(history_acc)/len(history_acc)*100, 1))
    print(history_acc)
    print('model ACC: ', round(sum(history_acc)/len(history_acc)*100, 1), '%')
    
def report(modelsssss):
    for model in modelsssss:
        print(model.summary())
def report_plot_experiments():
    print(accuraices)
    objects = ('Arch1' , 'Arch2' , 'Arch3' ,'Arch4')
    y_pos = np.arange(len(objects))
    performance = accuraices
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracies')
    plt.title('Archs')
        

experiment([450, 220, 110, 50, 20, 10], 1 ,
           ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'] , [10 , 10])


experiment([220, 50, 10], 2,
           ['relu', 'relu', 'softmax'] , [5,5])

experiment([180, 30, 10], 3,
           ['relu', 'relu', 'softmax'] , [2,2])


experiment([100, 10] , 4 ,
           ['relu', 'softmax'] , [2,2] )






report(models)
report_plot_experiments()
