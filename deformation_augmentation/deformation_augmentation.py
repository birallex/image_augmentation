from keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2

def deformate_image(mat, kx_coefficient, ky_coefficient):
    height, width = mat.shape[:2]
    M_mat = np.float32([[kx_coefficient, 0, 0], [0, ky_coefficient, 0]])
    deformated_mat = cv2.warpAffine(mat, M_mat, (width, height))
    return deformated_mat

def deformate_and_crop_image(mat, kx_coefficient, ky_coefficient):
    height, width = mat.shape[:2]
    M_mat = np.float32([[kx_coefficient, 0, 0], [0, ky_coefficient, 0]])
    deformated_mat = cv2.warpAffine(mat, M_mat, (int(width*kx_coefficient), int(height*ky_coefficient)))
    return deformated_mat
    

if __name__ == "__main__":
    img = image.load_img("../input_image.jpg")
    image_matrix = image.img_to_array(img)
    output_img_matrix = deformate_image(image_matrix, 0.8, 0.35)
    output_image = image.array_to_img(output_img_matrix)

    output_image.save("deformation_augmentation_output.jpg")
