from keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2

def scaling_image(mat, coefficient):
    height, width = mat.shape[:2]
    M_mat = np.float32([[coefficient, 0, 0], [0, coefficient, 0]])
    #данная строка масштабирует, но не обрезает изображение / this line scales but does not crop the image 
    scaled_mat = cv2.warpAffine(mat, M_mat, (width, height))
    return scaled_mat

def scale_and_crop_image(mat, coefficient):
    height, width = mat.shape[:2]
    M_mat = np.float32([[coefficient, 0, 0], [0, coefficient, 0]])
    #следующая строка масштабирует и обрезает изображение согласно масштабу
    #the next line scales and crops the image to scale: 
    scaled_mat = cv2.warpAffine(mat, M_mat, (int(width*coefficient), int(height*coefficient)))
    return scaled_mat
    

if __name__ == "__main__":
    img = image.load_img("../input_image.jpg")
    image_matrix = image.img_to_array(img)
    output_img_matrix = scaling_image(image_matrix, 0.9)
    output_image = image.array_to_img(output_img_matrix)

    output_image.save("scale_augmentation_output.jpg")
