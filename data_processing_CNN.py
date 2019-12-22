import cv2 as cv
import os
import random
import csv
import pandas as pd
import numpy as np


WIDTH = 30
HEIGHT = 30

def create_csv_file(data):
    columns = []
    for i in range(len(data[0])-10):
        columns.append("pixel"+str(i+1))
    for i in range(10):
        columns.append("label"+str(i))
    with open('images_NN.csv' , 'w', newline = "") as f:
        write = csv.writer(f)
        write.writerow(columns)
        for i in range(len(data)):
            write.writerow(data[i])
            
#################################################################            

def one_hot_encoder(x):
    res = [0] * 10
    res[x] = 1
    return res

#################################################################

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
#################################################################

def resize_image(image):
    return cv.resize(image , (WIDTH , HEIGHT) )

#################################################################
    
def flatten(normalized_image , label ):
    flat_image = []
    for row in normalized_image:
        for pixel in row:
            flat_image += list(pixel)
    flat_image += label
    return flat_image

#################################################################
    
def read_RGB_image(image_path):
    img = cv.imread(image_path)
    return img

def convert_to_gray_image(img):
    image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return image_gray
    
#################################################################
        
def read_images_from_dir(dir_path):
    images = []
    for image_name in os.listdir(dir_path):
        images.append(read_RGB_image(dir_path+"/"+image_name))
    return images

#################################################################  
    
def read_images_from_dirs(dir_path):
     images = []
     for dir_name in os.listdir(dir_path):
         cur_images =  read_images_from_dir(dir_path+"/"+dir_name)
         cur_images_with_label = []
         for i in cur_images:
             cur_images_with_label.append((i, dir_name))
         images += cur_images_with_label
     return images
 
#################################################################
    
def normalize_images(images):
    avg_image = [[[0, 0, 0]] * images[0].shape[1] for i in range(images[0].shape[0])] 
    
    print(type(avg_image))
    for image in images:
        for i in range(image[0].shape[0]):
            for j in range(image[0].shape[1]):           
                avg_image[i][j][0] += float(image[0][i][j][0])
                avg_image[i][j][1] += float(image[0][i][j][1])
                avg_image[i][j][2] += float(image[0][i][j][2])
        """
        for i in range(image[0].shape[0]):
            for j in range(image[0].shape[1]):
                avg_image[i][j][0] = float(avg_image[i][j][0]/len(images)*1.0)
                avg_image[i][j][1] = float(avg_image[i][j][1]/len(images)*1.0)
                avg_image[i][j][2] = float(avg_image[i][j][2]/len(images)*1.0)
        """
    print(len( avg_image ))
    normalized_images = images # copy of the original data
    
    for k in range(len(images)):
        for i in range(images[k][0].shape[0]):
            for j in range(images[k][0].shape[1]):
                normalized_images[k][0][i][j][0] = float(normalized_images[k][0][i][j][0]) - float(avg_image[i][j][0])
                #normalized_images[k][0][i][j][0] /= 255.0
                normalized_images[k][0][i][j][1] = float(normalized_images[k][0][i][j][1]) -float(avg_image[i][j][1])
                #normalized_images[k][0][i][j][1] /= 255.0
                normalized_images[k][0][i][j][2] = float(normalized_images[k][0][i][j][2]) - float(avg_image[i][j][2])
                #normalized_images[k][0][i][j][2] /= 255.0
    return normalized_images

#################################################################
 
def collect_images(parent_dir_path):
    collected_images = []
    images = read_images_from_dirs(parent_dir_path)
    print("images : " , images[0][0].shape)
    for i in range(len(images)):
        images[i] = (resize_image(images[i][0]), images[i][1])
    normalized_images = normalize_images(images)
    for i in range(len(images)):
        output_label = one_hot_encoder(int(images[i][1]))
        flat_image = flatten(normalized_images[i][0], output_label)
        collected_images.append(flat_image)
    return collected_images

#################################################################
    
def split_data(data, train=.6, test=.2, validation=.2):
    random.shuffle(data)
    train_data = data[:int(train * len(data))]
    validation_data = data[int(train * len(data)):int((train+validation) * len(data))]
    test_data = data[int((train+validation) * len(data)):]
    return train_data, validation_data, test_data


#################################################################
    
def final_data():
    data = pd.read_csv("images_NN.csv")
    data = data.values.tolist()
    train_data, validation_data, test_data = split_data(data)
   
    return train_data , validation_data ,test_data

################################################################
"""
data = collect_images("Sign-Language-Digits-Dataset-master/Dataset")
train_data, validation_data, test_data = split_data(data)
print(train_data[0][900:])
"""
'''
train_data, validation_data, test_data = final_data()
with open('train.csv' , 'w', newline = "") as f:
        write = csv.writer(f)
        for i in range(len(train_data)):
            write.writerow(train_data[i])
'''
"""
data = collect_images("Sign-Language-Digits-Dataset-master/Dataset")    
create_csv_file(data)
"""
"""

images = read_images_from_dir("Sign-Language-Digits-Dataset-master/Dataset/0")
normalized_images = normalize_images(images)
print(normalized_images[0].shape)
"""

#collected_images = collect_images("Sign-Language-Digits-Dataset-master/Dataset")

images = read_images_from_dirs("Sign-Language-Digits-Dataset-master/Dataset")
print("first image" , images[0][0].shape)

"""
array1  = [[1 , 2 , 3] , [1, 2 ,3] , [1, 2 ,3] , [1, 2 ,3]]
array2  = [[1 , 2 , 3] , [1, 2 ,3] , [1, 2 ,3] , [1, 2 ,3]]
array3  = [[1 , 2 , 3] , [1, 2 ,3] , [1, 2 ,3] , [1, 2 ,3]]
array4  = [[1 , 2 , 3] , [1, 2 ,3] , [1, 2 ,3] , [1, 2 ,3]]
array = []
array +=[array1]
array +=[array2]
array +=[array3] 
array +=[array4]

"""


    
print(len(images[0[0,:]]))

#mean_images = np.mean(last_images)


"""

print("================================")
print("mean Shape : " , mean_images.shape)
images[:][0] -= mean_images
print("================================")
print("mean Shape : " , images[0][0])

print(images[0][1])
show_image(images[0][0])

#print(images[0][0])
print("================================")

#images /=255.0

#print(images[0])

"""