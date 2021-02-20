from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


def primitive_brightness(source_matrix, brightness):
    """loads cpu too much"""
    result = np.zeros(source_matrix.shape)
    for x in range(source_matrix.shape[0]):
        for y in range(source_matrix.shape[1]):
            r, g, b = source_matrix[x][y]

            red = int(r * brightness)
            red = min(255, max(0, red))

            green = int(g * brightness)
            green = min(255, max(0, green))

            blue = int(b * brightness)
            blue = min(255, max(0, blue))

            result[x][y] = red, green, blue 
    return result

def brightness(source_matrix, brightness):
    result = np.copy(source_matrix)
    result *= brightness
    result = np.clip(result, 0, 255) 
    return result


if __name__ == "__main__":
    img = image.load_img("../input_image.jpg")
    image_matrix = image.img_to_array(img)
    output_img_matrix = brightness(image_matrix, 1.6)
    output_image = image.array_to_img(output_img_matrix)
    output_image.save("brightness_augmentation_output.jpg")

