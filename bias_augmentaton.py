from keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2

def image_bias(matrix, x_bias, y_bias):
    num_rows, num_cols = matrix.shape[:2]
    translation_matrix = np.float32([[1, 0, x_bias], [0, 1, y_bias]])
    img_translation = cv2.warpAffine(matrix, translation_matrix, (num_cols, num_rows))
    return img_translation


if __name__ == "__main__":
    img = image.load_img("input_image.jpg")

    image_matrix = image.img_to_array(img)

    output_img_matrix = image_bias(image_matrix, 400, 200)
    output_image = image.array_to_img(output_img_matrix)

    output_image.save("bias_augmentation_output.jpg")