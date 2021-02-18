from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import random

def noise(source_matrix, noise_level):
    result = np.copy(source_matrix)
    for x in range(source_matrix.shape[0]):
        for y in range(source_matrix.shape[1]):
            if result[x][y].any() != 0:
                noise = random.randint(-noise_level, noise_level)
                result[x][y] += noise
    result = np.clip(result, 0, 255) 
    return result


if __name__ == "__main__":
    img = image.load_img("../input_image.jpg")
    image_matrix = image.img_to_array(img)
    output_img_matrix = noise(image_matrix, 100)
    output_image = image.array_to_img(output_img_matrix)
    output_image.save("noise_augmentation_output.jpg")