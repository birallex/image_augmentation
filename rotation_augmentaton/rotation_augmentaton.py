from keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2

def rotate_image(mat, angle):
    'Rotates the image matrix by a given angle'
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

if __name__ == "__main__":
    img = image.load_img("../input_image.jpg")
    image_matrix = image.img_to_array(img)

    output_img_matrix = rotate_image(image_matrix, 30)
    output_image = image.array_to_img(output_img_matrix)

    output_image.save("rotate_augmentation_output.jpg")