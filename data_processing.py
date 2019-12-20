import cv2 as cv
import os
import random


WIDTH = 30
HEIGHT = 30

def one_hot_encoder(x):
    res = [0] * 10
    res[x] = 1
    return res

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def resize_image(image):
    return cv.resize(image , (WIDTH , HEIGHT) )

def normalize_image(image):
    normalized_image = [[0] * image.shape[1] for i in range(image.shape[0])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            normalized_image[i][j] = round(1.0*image[i][j]/255, 2)
    return normalized_image

def flatten(normalized_image , label ):
    flat_image = []
    for i in normalized_image:
        flat_image += i
    flat_image += label
    return flat_image
    
def read_image(image_path):
        img = cv.imread(image_path, 0)
        return img
    
def read_images_from_dir(dir_path):
    images = []
    for image_name in os.listdir(dir_path):
        images.append(read_image(dir_path+"/"+image_name))
    return images
    
def read_images_from_dirs(dir_path):
     images = []
     for dir_name in os.listdir(dir_path):
         cur_images =  read_images_from_dir(dir_path+"/"+dir_name)
         cur_images_with_label = []
         for i in cur_images:
             cur_images_with_label.append((i, dir_name))
         images += cur_images_with_label
     return images

def collect_images(parent_dir_path):
    collected_images = []
    images = read_images_from_dirs(parent_dir_path)
    for i in range(len(images)):
        cur_image = resize_image(images[i][0])
        normalized_image= normalize_image(cur_image)
        output_label = one_hot_encoder(int(images[i][1]))
        flat_image = flatten(normalized_image, output_label)
        collected_images.append(flat_image)
    return collected_images

def split_data(data, train=.6, test=.2, validation=.2):
    random.shuffle(data)
    train_data = data[:int(train * len(data))]
    validation_data = data[int(train * len(data)):int((train+validation) * len(data))]
    test_data = data[int((train+validation) * len(data)):]
    return train_data, validation_data, test_data

data = collect_images("Sign-Language-Digits-Dataset-master/Dataset")
#print(data[0])
#print(len(data))
#print(len(data[0]))
train_data, validation_data, test_data = split_data(data)

print(train_data[0])
print(train_data[1])
print(train_data[2])
print("******************************************");
print(len(train_data))
print("=================================================================");
print(validation_data[0])
print(validation_data[1])
print(validation_data[2])
print("******************************************");
print(len(validation_data))
print("=================================================================");
print(test_data[0])
print(test_data[1])
print(test_data[2])
print("******************************************");
print(len(test_data))
print("=================================================================");





