from data_processing import final_data , WIDTH , HEIGHT
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report

image_size = WIDTH*HEIGHT
models = []
model_losses = []
accuraices = []

def extract_train_valid_test():
    train_data , validation_data ,test_data = final_data()

    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test =  list()
    X_validation = list()
    Y_validation = list()
    
    for i in range(len(train_data)):
        X_train.append(train_data[i][:image_size])
        Y_train.append(train_data[i][image_size:])
    Y_train = [list(map(int, i)) for i in Y_train]
    for i in range(len(validation_data)):
        X_validation.append(validation_data[i][:image_size])
        Y_validation.append(validation_data[i][image_size:])
    Y_validation = [list(map(int, i)) for i in Y_validation]
    for i in range(len(test_data)):
        X_test.append(test_data[i][:image_size])
        Y_test.append(test_data[i][image_size:])
    Y_test = [list(map(int, i)) for i in Y_test]  
    return X_train, Y_train , X_validation , Y_validation , X_test , Y_test

def build_model(n_layers, activ_func_layers):
    model = keras.Sequential()
    model.add(keras.layers.Dense(n_layers[0], 
                                 activation=activ_func_layers[0], 
                                 input_shape=(image_size,)))
    for i in range(1, len(n_layers)):
        model.add(keras.layers.Dense(n_layers[i], 
                                     activation=activ_func_layers[i]))
    #model.summary()    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


def k_fold_arch_NN(n_layers, activ_func_layers, k_fold=3):
    history_acc = []
    history_loss = []
    for i in range(k_fold):
        X_train, Y_train , X_validation , Y_validation , X_test , Y_test = extract_train_valid_test()
        model = build_model(n_layers, activ_func_layers)
        history = model.fit(X_train , Y_train ,epochs = 20 , validation_data=(X_validation , Y_validation) , batch_size=64)
        test_loss, test_acc = model.evaluate(X_test , Y_test)        
        Y_pred = model.predict(X_test)
        y_labels_predicted = []
        y_labels_actual = []
        for i in range(len(Y_pred)):
            y_labels_predicted.append(np.argmax(Y_pred[i]))
            y_labels_actual.append(np.argmax(Y_test[i]))
        print(classification_report(y_labels_actual , y_labels_predicted))
        keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        Y_pred = [int(np.argmax(i)) for i in Y_pred]
        history_acc.append(history.history['accuracy'][-1]) #last epoch acc
    models.append(model)
    return history_acc

def experiment(n_layers, activ_func_layers, k_fold=3):
    history_acc = k_fold_arch_NN(n_layers, activ_func_layers, k_fold)
    accuraices.append(round(sum(history_acc)/len(history_acc)*100, 1))
    print(history_acc)
    print('model ACC: ', round(sum(history_acc)/len(history_acc)*100, 1), '%')

def report():
    print(len(models))
    print(len(accuraices))
    for i in range(len(models)):
        with open('report_NN.txt','a') as out:
        # Pass the file handle in as a lambda function to make it callable
            models[i].summary(print_fn=lambda x: out.write(x + '\n'))
            out.write("Train Accuracy : ")
            out.write( str(accuraices[i])+ "%")
            print(models[i].summary())
        
def report_plot_experiments():
    print(accuraices)
    objects = ('Arch1' , 'Arch2' , 'Arch3' ,'Arch4')
    y_pos = np.arange(len(objects))
    performance = accuraices
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracies')
    plt.title('Archs')



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arch 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
experiment([450, 220, 110, 50, 20, 10],
           ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'])

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arch 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
experiment([220, 50, 10],
           ['relu', 'relu', 'softmax'])

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arch 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
experiment([180, 30, 10],
           ['relu', 'relu', 'softmax'])

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arch 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
experiment([100, 10],
           ['relu', 'softmax'])


report()
#report_plot_experiments()